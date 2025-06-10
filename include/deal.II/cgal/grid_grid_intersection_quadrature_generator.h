#ifndef dealii_grid_grid_intersection_quadrature_generator_h
#define dealii_grid_grid_intersection_quadrature_generator_h

#include <deal.II/base/config.h>

#include <deal.II/fe/mapping_q.h> //think of this and mapping header

#include <deal.II/distributed/tria.h> //think of this and normal tria header

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#ifdef DEAL_II_WITH_CGAL
#   include "polygon_2.h"
#   include <CGAL/Polygon_with_holes_2.h>
#   include <CGAL/Delaunay_triangulation_2.h>
#   include <CGAL/intersections.h>
#   include <CGAL/Boolean_set_operations_2.h>

#   include <deal.II/cgal/surface_mesh.h>
#   include <deal.II/cgal/utilities.h>
#   include <CGAL/Polygon_mesh_processing/clip.h>
#   include <CGAL/Side_of_triangle_mesh.h>

//output
#   include <CGAL/IO/output_to_vtu.h>
#   include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#   include <CGAL/IO/VTK.h>
#   include <deal.II/base/timer.h>


DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
    template <int dim>
    class GridGridIntersectionQuadratureGenerator
    {
        using K = CGAL::Exact_predicates_exact_constructions_kernel;

        // 3D
        using CGALPoint = CGAL::Point_3<K>;
        using CGALTriangulation = CGAL::Triangulation_3<K>;

        // 2D
        using CGALPoint2 = CGAL::Point_2<K>;
        using CGALPolygon = CGAL::Polygon_2<K>;
        using CGALPolygonWithHoles = CGAL::Polygon_with_holes_2<K>;
        using Iso_rectangle_2 = CGAL::Iso_rectangle_2<K>;
        using CGALSegment2 = CGAL::Segment_2<K>;
        using Triangulation2 = CGAL::Delaunay_triangulation_2<K>;

    public:
        GridGridIntersectionQuadratureGenerator()
            : mapping(nullptr), quadrature_order(0), boolean_operation(BooleanOperation::compute_intersection)
        {
            Assert(dim == 2 || dim == 3,
                   ExcMessage("GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
        };

        GridGridIntersectionQuadratureGenerator(
            const MappingQ<dim> &mapping_in,
            unsigned int quadrature_order_in,
            BooleanOperation boolean_operation_in);

        void reinit(
            const MappingQ<dim> &mapping_in,
            unsigned int quadrature_order_in,
            BooleanOperation boolean_operation_in);

        void clear();
        
        template<typename TriangulationType>
        void reclassify(const TriangulationType &tria_unfitted_in,
                        const TriangulationType &tria_fitted_in);

        void generate(const typename Triangulation<dim>::cell_iterator &cell);

        void generate_dg_face(const typename Triangulation<dim>::cell_iterator &cell, unsigned int face_index);

        NonMatching::ImmersedSurfaceQuadrature<dim> get_surface_quadrature() const;

        Quadrature<dim> get_inside_quadrature() const;

        Quadrature<dim - 1> get_inside_quadrature_dg_face() const;

        NonMatching::LocationToLevelSet location_to_geometry(unsigned int cell_index) const;
        NonMatching::LocationToLevelSet location_to_geometry(
            const typename Triangulation<dim>::cell_iterator &cell) const;

        void output_fitted_mesh() const;

    private:

        const MappingQ<dim> *mapping;
        unsigned int quadrature_order;
        BooleanOperation boolean_operation;

        CGALPolygon fitted_2D_mesh;
        CGAL::Surface_mesh<CGALPoint> fitted_surface_mesh;

        Quadrature<dim> quad_cells;
        NonMatching::ImmersedSurfaceQuadrature<dim> quad_surface;
        std::vector<NonMatching::LocationToLevelSet> location_to_geometry_vec;
        Quadrature<dim - 1> quad_dg_face;
    };

    template <int dim>
    GridGridIntersectionQuadratureGenerator<dim>::GridGridIntersectionQuadratureGenerator(
        const MappingQ<dim> &mapping_in,
        unsigned int quadrature_order_in,
        BooleanOperation boolean_operation_in)
        : mapping(&mapping_in), quadrature_order(quadrature_order_in), boolean_operation(boolean_operation_in)
    {
        Assert(dim == 2 || dim == 3, ExcMessage("GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
        Assert(boolean_operation_in == BooleanOperation::compute_intersection || boolean_operation_in == BooleanOperation::compute_difference,
            ExcMessage("Union and corerefinement not implemented"));
    }

    template <int dim>
    void GridGridIntersectionQuadratureGenerator<dim>::reinit(
        const MappingQ<dim> &mapping_in,
        unsigned int quadrature_order_in,
        BooleanOperation boolean_operation_in)
    {
        mapping = &mapping_in;
        quadrature_order = quadrature_order_in;
        boolean_operation = boolean_operation_in;
        Assert(boolean_operation_in == BooleanOperation::compute_intersection || boolean_operation_in == BooleanOperation::compute_difference,
            ExcMessage("Union and corerefinement not implemented"));
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<2>::clear()
    {
        quad_cells = Quadrature<2>();
        quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>();
        location_to_geometry_vec.clear();
        fitted_2D_mesh.clear();
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<3>::clear()
    {
        quad_cells = Quadrature<3>();
        quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>();
        location_to_geometry_vec.clear();
        fitted_surface_mesh.clear();
    }

    // The classification inside is only valid if no vertex is on the
    // boundary. Technically if one vertex is on the boundary the cell could still
    // be completely inside but a conservatice approach is choosen.
    // Since we only check for edges we guess it might intersect.
    //-> in this case we will generate a volume integral over whole cell!
    template<>
    template <typename TriangulationType>
    void GridGridIntersectionQuadratureGenerator<2>::reclassify(
        const TriangulationType &tria_unfitted, const TriangulationType &tria_fitted)
    {
        Timer timer; // debug
        fitted_2D_mesh.clear();
        CGALWrappers::dealii_tria_to_cgal_polygon(tria_fitted, fitted_2D_mesh);
        timer.stop();
        std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n"; // debug
        std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n"; // debug

        Assert(fitted_2D_mesh.is_simple(), ExcMessage("Polygon not simple"));
        Assert(fitted_2D_mesh.is_counterclockwise_oriented(), ExcMessage("Polygon not oriented"));

        location_to_geometry_vec.clear();
        location_to_geometry_vec.reserve(tria_unfitted.n_active_cells());

        CGAL::Bounded_side inside_domain;
        if(boolean_operation == BooleanOperation::compute_intersection)
        {
            inside_domain = CGAL::ON_BOUNDED_SIDE;
        }
        else if(boolean_operation == BooleanOperation::compute_difference)
        {
            inside_domain = CGAL::ON_UNBOUNDED_SIDE;
        }

        // now find out if inside or not
        for (const auto &cell : tria_unfitted.active_cell_iterators())
        {
            CGALPolygon polygon_cell;
            CGALWrappers::dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

            // requires smooth boundaries otherwise it might make mistakes
            // since it only checks for edge nodes
            // Assert checks for this in debug mode but this is very expensive
            unsigned int inside_count = 0;
            for (unsigned int i = 0; i < cell->n_vertices(); i++)
            {
                auto result = CGAL::bounded_side_2(fitted_2D_mesh.begin(), fitted_2D_mesh.end(),
                                                   CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(cell->vertex(i)));
                inside_count += (result == inside_domain);
            }
            if (inside_count == 0) //case 1: all vertices outside or on boundary: not considered (outside)
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::outside);
                //Assert(!CGAL::do_intersect(fitted_2D_mesh, polygon_cell), ExcMessage("cell classified as outside although intersected"));
            }
            else if (inside_count == cell->n_vertices()) //case 2: all vertices inside: considered (inside)
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::inside);
            }
            else //case 3: at least one vertex inside and at least one on boundary or outside
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::intersected);
            }
            //Note: construction of case 1 and 3 make sure that if two vertices are on the boundary the cell
            // inside is considered cut and takes acount for the face integral. The cell outside also has 
            // vertices on the boundary but is ignored because then the boundary integral would be performed twice
        }
    }

    template<>
    template <typename TriangulationType>
    void GridGridIntersectionQuadratureGenerator<3>::reclassify(
        const TriangulationType &tria_unfitted, const TriangulationType &tria_fitted)
    {
        fitted_surface_mesh.clear();
        CGALWrappers::dealii_tria_to_cgal_surface_mesh<CGALPoint>(
            tria_fitted, fitted_surface_mesh);
        CGAL::Polygon_mesh_processing::triangulate_faces(fitted_surface_mesh);
        location_to_geometry_vec.clear();
        location_to_geometry_vec.reserve(tria_unfitted.n_active_cells());

        CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K>
            inside_test(fitted_surface_mesh);
        
        CGAL::Bounded_side inside_domain;
        if(boolean_operation == BooleanOperation::compute_intersection)
        {
            inside_domain = CGAL::ON_BOUNDED_SIDE;
        }
        else if(boolean_operation == BooleanOperation::compute_difference)
        {
            inside_domain = CGAL::ON_UNBOUNDED_SIDE;
        }

        for (const auto &cell : tria_unfitted.active_cell_iterators())
        {
            unsigned int inside_count = 0;
            for (size_t i = 0; i < cell->n_vertices(); i++)
            {
                auto result = inside_test(CGALWrappers::dealii_point_to_cgal_point<CGALPoint, 3>(cell->vertex(i)));
                inside_count += (result == inside_domain);
            }

            if (inside_count == 0)
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::outside);
            }
            else if (inside_count == cell->n_vertices())
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::inside);
            }
            else
            {
                location_to_geometry_vec.push_back(NonMatching::LocationToLevelSet::intersected);
            }
        }
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<2>::generate(
        const typename Triangulation<2>::cell_iterator &cell)
    {
        CGALPolygon polygon_cell;
        CGALWrappers::dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

        std::vector<CGALPolygonWithHoles> polygon_out_vec;

        CGALWrappers::compute_boolean_operation(polygon_cell, fitted_2D_mesh, boolean_operation, polygon_out_vec);

        // quadrature area in a cell could be split into two polygons
        // code should be able to handle this. But occurence is not expected for
        // smooth boundaries as needed for reclassify (+not tested enough)
        Assert(polygon_out_vec.size() == 1, ExcMessage("Not a single polygon with holes, disconnected domain!!"));

        std::vector<std::array<dealii::Point<2>, 3>> vec_of_simplices;
        for (size_t i = 0; i < polygon_out_vec.size(); i++)
        {
            // no usecase where a quadrature area in a cell should be a polygon with holes
            Assert(!polygon_out_vec[i].has_holes(), ExcMessage("The Polygon has holes"));

            Triangulation2 tria;
            tria.insert(polygon_out_vec[i].outer_boundary().vertices_begin(), polygon_out_vec[i].outer_boundary().vertices_end());

            // Extract simplices and construct quadratures
            for (const auto &face : tria.finite_face_handles())
            {
                std::array<dealii::Point<2>, 3> simplex;
                std::array<dealii::Point<2>, 3> unit_simplex;
                for (unsigned int i = 0; i < 3; ++i)
                {
                    simplex[i] =
                        CGALWrappers::cgal_point_to_dealii_point<2>(face->vertex(i)->point());
                }
                mapping->transform_points_real_to_unit_cell(cell, simplex, unit_simplex);
                vec_of_simplices.push_back(unit_simplex);
            }
        }
        quad_cells = QGaussSimplex<2>(quadrature_order).mapped_quadrature(vec_of_simplices);

        // surface quadrature
        std::vector<Point<2>> quadrature_points;
        std::vector<double> quadrature_weights;
        std::vector<Tensor<1, 2>> normals;
        for(const auto &edge_boundary : fitted_2D_mesh.edges())
        {
            auto bs_1 = CGAL::bounded_side_2(
                            polygon_cell.begin(), polygon_cell.end(),
                            edge_boundary.source());
            auto bs_2 = CGAL::bounded_side_2(
                            polygon_cell.begin(), polygon_cell.end(),
                            edge_boundary.target());

            std::array<Point<2>, 2> segment;
            if( bs_1 != CGAL::ON_UNBOUNDED_SIDE && 
                bs_2 != CGAL::ON_UNBOUNDED_SIDE) //add directly
            {
                segment[0] = CGALWrappers::cgal_point_to_dealii_point<2>(edge_boundary.source());
                segment[1] = CGALWrappers::cgal_point_to_dealii_point<2>(edge_boundary.target());
            }
            else if(bs_1 == CGAL::ON_BOUNDARY || bs_2 == CGAL::ON_BOUNDARY) //only point on boundary
            {
                continue;
            }
            else if(bs_1 == CGAL::ON_BOUNDED_SIDE) //find missing intersection point
            {
                for(const auto &edge_cell : polygon_cell.edges())
                {
                    auto result = CGAL::intersection(edge_cell, edge_boundary);
                    if(result)
                    {
                        if(const CGALPoint2* point = boost::get<CGALPoint2>(&*result))
                        {
                            segment[0] = CGALWrappers::cgal_point_to_dealii_point<2>(edge_boundary.source());
                            segment[1] = CGALWrappers::cgal_point_to_dealii_point<2>(*point);
                            break;
                        }
                    }
                }
            }
            else if(bs_2 == CGAL::ON_BOUNDED_SIDE) //find missing intersection point
            {
                for(const auto &edge_cell : polygon_cell.edges())
                {
                    auto result = CGAL::intersection(edge_cell, edge_boundary);
                    if(result)
                    {
                        if(const CGALPoint2* point = boost::get<CGALPoint2>(&*result))
                        {
                            segment[0] = CGALWrappers::cgal_point_to_dealii_point<2>(*point);
                            segment[1] = CGALWrappers::cgal_point_to_dealii_point<2>(edge_boundary.target());
                            break;
                        }
                    }
                }
            }
            else //find two missing points or nothing
            {
                int i = 0;
                for(const auto &edge_cell : polygon_cell.edges())
                {
                    auto result = CGAL::intersection(edge_cell, edge_boundary);
                    if(result)
                    {
                        if(const CGALPoint2* point = boost::get<CGALPoint2>(&*result))
                        {
                            segment[i] = CGALWrappers::cgal_point_to_dealii_point<2>(*point);
                            i += 1;
                            if(i == 2)
                                break;
                        }
                    }
                }

                if(i < 2)
                    continue;
            }

            std::array<dealii::Point<2>, 2> unit_segment;
            mapping->transform_points_real_to_unit_cell(cell, segment, unit_segment);
            auto quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation(unit_segment);
            auto points = quadrature.get_points();
            auto weights = quadrature.get_weights();

            // compute normals
            Tensor<1, 2> normal = unit_segment[1] - unit_segment[0];
            std::swap(normal[0], normal[1]);

            auto test_point = 0.5 * (unit_segment[1] + unit_segment[0]) + normal * 1;
            auto flip = CGAL::bounded_side_2(
                            polygon_cell.begin(), polygon_cell.end(),
                            CGALPoint2(test_point[0], test_point[1]));
            if(boolean_operation == BooleanOperation::compute_intersection &&
                flip == CGAL::ON_BOUNDED_SIDE)
            {
                normal *= -1;
            }
            else if(boolean_operation == BooleanOperation::compute_difference &&
                    flip == CGAL::ON_UNBOUNDED_SIDE)
            {
                normal *= -1;
            }
            normal /= normal.norm();


            quadrature_points.insert(quadrature_points.end(), points.begin(), points.end());
            quadrature_weights.insert(quadrature_weights.end(), weights.begin(), weights.end());
            normals.insert(normals.end(), quadrature.size(), normal);

        }
        quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>(quadrature_points, quadrature_weights, normals);
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<3>::generate(
        const typename Triangulation<3>::cell_iterator &cell)
    {
        CGAL::Surface_mesh<CGALPoint> fitted_surface_mesh_copy(fitted_surface_mesh);
        CGAL::Surface_mesh<CGALPoint> surface_cell;
        CGALWrappers::dealii_cell_to_cgal_surface_mesh(cell, *mapping, surface_cell);
        CGAL::Polygon_mesh_processing::triangulate_faces(surface_cell);

        { // not needed for surface calculation
            CGAL::Surface_mesh<CGALPoint> out_surface;
            CGALWrappers::compute_boolean_operation(surface_cell, fitted_surface_mesh_copy,
                                                    boolean_operation, out_surface);

            // Fill triangulation with vertices from surface mesh
            CGALTriangulation tria;
            tria.insert(out_surface.points().begin(), out_surface.points().end());

            // Extract simplices and construct quadratures
            std::vector<std::array<dealii::Point<3>, 4>> vec_of_simplices;
            for (const auto &face : tria.finite_cell_handles())
            {
                std::array<dealii::Point<3>, 4> simplex;
                std::array<dealii::Point<3>, 4> unit_simplex;
                for (unsigned int i = 0; i < 4; ++i)
                {
                    simplex[i] =
                        CGALWrappers::cgal_point_to_dealii_point<3>(face->vertex(i)->point());
                }
                mapping->transform_points_real_to_unit_cell(cell, simplex, unit_simplex);
                vec_of_simplices.push_back(unit_simplex);
            }
            quad_cells = QGaussSimplex<3>(quadrature_order).mapped_quadrature(vec_of_simplices);
        }
        // seems to work but keep in ming maybe need to use new copys
        bool manifold = CGAL::Polygon_mesh_processing::clip(fitted_surface_mesh_copy, surface_cell);
        Assert(manifold, ExcMessage("The clipped surface mesh is not a manifold"));

        CGAL::Polygon_mesh_processing::remove_degenerate_faces(fitted_surface_mesh_copy);

        std::vector<Point<3>> quadrature_points;
        std::vector<double> quadrature_weights;
        std::vector<Tensor<1, 3>> normals;
        double ref_area = std::pow(cell->minimum_vertex_distance(), 2) * 0.0000001;
        for (const auto &face : fitted_surface_mesh_copy.faces())
        {
            if (CGAL::abs(CGAL::Polygon_mesh_processing::face_area(face, fitted_surface_mesh_copy)) < ref_area)
            {
                continue;
            }
            std::array<Point<3>, 3> simplex;
            std::array<Point<3>, 3> unit_simplex;
            int i = 0;
            for (const auto &vertex : CGAL::vertices_around_face(fitted_surface_mesh_copy.halfedge(face), fitted_surface_mesh_copy))
            {
                simplex[i] = CGALWrappers::cgal_point_to_dealii_point<3>(fitted_surface_mesh_copy.point(vertex));
                i += 1;
            }
            // compute quadrature and fill vectors
            mapping->transform_points_real_to_unit_cell(cell, simplex, unit_simplex);
            auto quadrature = QGaussSimplex<2>(quadrature_order).compute_affine_transformation(unit_simplex);
            auto points = quadrature.get_points();
            auto weights = quadrature.get_weights();
            quadrature_points.insert(quadrature_points.end(), points.begin(), points.end());
            quadrature_weights.insert(quadrature_weights.end(), weights.begin(), weights.end());

            const Tensor<1, 3> v1 = simplex[2] - simplex[1];
            const Tensor<1, 3> v2 = simplex[0] - simplex[1];
            Tensor<1, 3> normal = cross_product_3d(v1, v2);
            normal /= normal.norm();
            normals.insert(normals.end(), quadrature.size(), normal);
        }
        quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>(quadrature_points, quadrature_weights, normals);

        if (quadrature_weights.empty())
            std::cout << "small intersection ignored this should not happen to much" << std::endl;
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<2>::generate_dg_face(
        const typename Triangulation<2>::cell_iterator &cell,
        unsigned int face_index)
    {
        auto face = cell->face(face_index);
        auto p_1 = CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(0));
        auto p_2 = CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(1));

        // if on boundary no dg flux!
        if( CGAL::bounded_side_2(fitted_2D_mesh.begin(), fitted_2D_mesh.end(),p_1) == CGAL::ON_BOUNDARY &&
            CGAL::bounded_side_2(fitted_2D_mesh.begin(), fitted_2D_mesh.end(),p_2) == CGAL::ON_BOUNDARY)
        {
            quad_dg_face = Quadrature<1>();
            return;
        }

        CGALPolygon polygon_cell;
        CGALWrappers::dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

        std::vector<CGALPolygonWithHoles> polygon_out_vec;
        CGALWrappers::compute_boolean_operation(polygon_cell, fitted_2D_mesh, BooleanOperation::compute_intersection, polygon_out_vec);

        Assert(polygon_out_vec.size() == 1, ExcMessage("Not a single polygon with holes, disconnected domain!!"));


        std::vector<Point<1>> quadrature_points;
        std::vector<double> quadrature_weights;
        for (size_t i = 0; i < polygon_out_vec.size(); i++)
        {
            Assert(!polygon_out_vec[i].has_holes(), ExcMessage("The Polygon has holes"));

            for (const auto &edge_cut : polygon_out_vec[i].outer_boundary().edges())
            {
                auto p_cut_1 = edge_cut.source();
                auto p_cut_2 = edge_cut.target();
                bool on_boundary_p_1 = 
                        CGAL::bounded_side_2(fitted_2D_mesh.begin(), fitted_2D_mesh.end(),p_cut_1) == CGAL::ON_BOUNDARY;
                bool on_boundary_p_2 =
                        CGAL::bounded_side_2(fitted_2D_mesh.begin(), fitted_2D_mesh.end(),p_cut_2) == CGAL::ON_BOUNDARY;
                // test if both endpoints are on on cell face
                if (!(on_boundary_p_1 && on_boundary_p_2))
                {
                    std::array<Point<2>, 2> face_points_unit;
                    // only linear mapping!!!
                    mapping->transform_points_real_to_unit_cell(cell,
                                                                {{CGALWrappers::cgal_point_to_dealii_point<2>(p_cut_1),
                                                                  CGALWrappers::cgal_point_to_dealii_point<2>(p_cut_1)}},
                                                                face_points_unit);

                    // only for quadrilateals so far
                    Quadrature<1> quadrature;
                    if(cell->reference_cell() == ReferenceCells::Quadrilateral){

                        if (face_index == 0 || face_index == 1){
                            quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation({{Point<1>(face_points_unit[0][1]), Point<1>(face_points_unit[1][1])}});
                        }else{
                            quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation({{Point<1>(face_points_unit[0][0]), Point<1>(face_points_unit[1][0])}});
                        }

                    }else if(cell->reference_cell() == ReferenceCells::Triangle){

                        //still need to test. Assume so far:
                        //     v2
                        //     | \
                        //  f0 |   \ f2
                        //     |     \
                        //    v0 ----- v1
                        //         f1
                        if(face_index == 0){
                            quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation({{Point<1>(face_points_unit[0][1]), Point<1>(face_points_unit[1][1])}});
                        }else if(face_index == 1){
                            quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation({{Point<1>(face_points_unit[0][0]), Point<1>(face_points_unit[1][0])}});
                        }else{
                            double distance1 = face_points_unit[0].distance(cell->vertex(1));
                            double distance2 = face_points_unit[0].distance(face_points_unit[1]);
                            quadrature = QGaussSimplex<1>(quadrature_order).compute_affine_transformation({{Point<1>(distance1), Point<1>(distance2)}});
                        }

                    }

                    auto points = quadrature.get_points();
                    auto weights = quadrature.get_weights();
                    quadrature_points.insert(quadrature_points.end(), points.begin(), points.end());
                    quadrature_weights.insert(quadrature_weights.end(), weights.begin(), weights.end());
                    // dont break for loop for the case where the face is split into two or more segments
                }
            }
        }
        quad_dg_face = Quadrature<1>(quadrature_points, quadrature_weights);
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<3>::generate_dg_face(const typename Triangulation<3>::cell_iterator &cell, unsigned int face_index)
    {
        Assert(false, ExcMessage("dg face generation only supports 2D so far"));
    }

    template <int dim>
    NonMatching::ImmersedSurfaceQuadrature<dim>
    GridGridIntersectionQuadratureGenerator<dim>::get_surface_quadrature() const
    {
        return quad_surface;
    }

    template <int dim>
    Quadrature<dim>
    GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature() const
    {
        return quad_cells;
    }

    template <int dim>
    Quadrature<dim - 1>
    GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature_dg_face() const
    {
        return quad_dg_face;
    }

    template <int dim>
    NonMatching::LocationToLevelSet
    GridGridIntersectionQuadratureGenerator<dim>::location_to_geometry(
        unsigned int cell_index) const
    {
        return location_to_geometry_vec[cell_index];
    }

    template <int dim>
    NonMatching::LocationToLevelSet
    GridGridIntersectionQuadratureGenerator<dim>::location_to_geometry(
        const typename Triangulation<dim>::cell_iterator &cell) const
    {
        return location_to_geometry_vec[cell->active_cell_index()];
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<2>::output_fitted_mesh() const
    {
        std::string filename = "fitted_polygon.vtu";
        std::ofstream file(filename);
        if (!file)
        {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }

        const std::size_t n = fitted_2D_mesh.size();

        file << R"(<?xml version="1.0"?>)" << "\n";
        file << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)" << "\n";
        file << R"(  <UnstructuredGrid>)" << "\n";
        file << R"(    <Piece NumberOfPoints=")" << n << R"(" NumberOfCells="1">)" << "\n";

        // Points section
        file << R"(      <Points>)" << "\n";
        file << R"(        <DataArray type="Float64" NumberOfComponents="3" format="ascii">)" << "\n";

        for (const auto &p : fitted_2D_mesh.container())
        {
            file << p.x() << " " << p.y() << " 0 ";
        }
        file << "\n";

        file << R"(        </DataArray>)" << "\n";
        file << R"(      </Points>)" << "\n";

        // Cells section
        // Connectivity: indices of vertices in order
        file << R"(      <Cells>)" << "\n";

        // Connectivity
        file << R"(        <DataArray type="Int32" Name="connectivity" format="ascii">)";
        for (std::size_t i = 0; i < n; ++i)
        {
            file << i << " ";
        }
        file << R"(</DataArray>)" << "\n";

        // Offsets: cumulative count of vertices after each cell
        // Here only one cell with n vertices
        file << R"(        <DataArray type="Int32" Name="offsets" format="ascii">)";
        file << n << R"(</DataArray>)" << "\n";

        // Types: VTK cell type for polygon is 7
        // (See https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
        file << R"(        <DataArray type="UInt8" Name="types" format="ascii">7</DataArray>)" << "\n";

        file << R"(      </Cells>)" << "\n";

        file << R"(    </Piece>)" << "\n";
        file << R"(  </UnstructuredGrid>)" << "\n";
        file << R"(</VTKFile>)" << "\n";

        file.close();
    }

    template <>
    void GridGridIntersectionQuadratureGenerator<3>::output_fitted_mesh() const
    {
        CGAL::IO::write_polygon_mesh("fitted_surface_mesh.stl", fitted_surface_mesh);
    }

}

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif
#endif