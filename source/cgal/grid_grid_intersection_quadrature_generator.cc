// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/cgal/grid_grid_intersection_quadrature_generator.h>

DEAL_II_NAMESPACE_OPEN

#ifndef DOXYGEN
// Template implementations
namespace CGALWrappers
{
  template <int dim>
  GridGridIntersectionQuadratureGenerator<
    dim>::GridGridIntersectionQuadratureGenerator()
    : mapping(nullptr)
    , quadrature_order(0)
    , boolean_operation(BooleanOperation::compute_intersection)
  {
    Assert(
      dim == 2 || dim == 3,
      ExcMessage(
        "GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
  };

  template <int dim>
  GridGridIntersectionQuadratureGenerator<dim>::
    GridGridIntersectionQuadratureGenerator(
      const MappingQ<dim> &mapping_in,
      unsigned int         quadrature_order_in,
      BooleanOperation     boolean_operation_in)
    : mapping(&mapping_in)
    , quadrature_order(quadrature_order_in)
    , boolean_operation(boolean_operation_in)
  {
    Assert(
      dim == 2 || dim == 3,
      ExcMessage(
        "GridGridIntersectionQuadratureGenerator only supports 2D and 3D"));
    Assert(boolean_operation_in == BooleanOperation::compute_intersection ||
             boolean_operation_in == BooleanOperation::compute_difference,
           ExcMessage("Union and corerefinement not implemented"));
  }

  template <int dim>
  void
  GridGridIntersectionQuadratureGenerator<dim>::reinit(
    const MappingQ<dim> &mapping_in,
    unsigned int         quadrature_order_in,
    BooleanOperation     boolean_operation_in)
  {
    mapping           = &mapping_in;
    quadrature_order  = quadrature_order_in;
    boolean_operation = boolean_operation_in;
    Assert(boolean_operation_in == BooleanOperation::compute_intersection ||
             boolean_operation_in == BooleanOperation::compute_difference,
           ExcMessage("Union and corerefinement not implemented"));
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::clear()
  {
    quad_cells   = Quadrature<2>();
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>();
    location_to_geometry_vec.clear();
    fitted_surface_mesh.clear();
    fitted_2D_mesh.clear();
    quad_dg_face_vec.clear();
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::clear()
  {
    quad_cells   = Quadrature<3>();
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>();
    location_to_geometry_vec.clear();
    fitted_surface_mesh.clear();
    fitted_2D_mesh.clear();
    quad_dg_face_vec.clear();
  }

  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<2>::setup_domain_boundary(
    const TriangulationType &tria_fitted_in)
  {
    fitted_2D_mesh.clear();
    dealii_tria_to_cgal_polygon(tria_fitted_in, fitted_2D_mesh);

    Assert(fitted_2D_mesh.is_simple(), ExcMessage("Polygon not simple"));
    Assert(fitted_2D_mesh.is_counterclockwise_oriented(),
           ExcMessage("Polygon not oriented"));
  }

  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<3>::setup_domain_boundary(
    const TriangulationType &tria_fitted_in)
  {
    fitted_surface_mesh.clear();
    dealii_tria_to_cgal_surface_mesh<CGALPoint>(tria_fitted_in,
                                                fitted_surface_mesh);
    CGAL::Polygon_mesh_processing::triangulate_faces(fitted_surface_mesh);

    Assert(fitted_surface_mesh.is_valid(),
           ExcMessage("The CGAL mesh is not valid"));
    Assert(CGAL::is_closed(fitted_surface_mesh),
           ExcMessage("The CGAL mesh is not closed"));
    Assert(CGAL::Polygon_mesh_processing::is_outward_oriented(
             fitted_surface_mesh),
           ExcMessage(
             "The normal vectors of the CGAL mesh are not oriented outwards"));
  }

  // The classification inside is only valid if no vertex is on the
  // boundary. Technically if one vertex is on the boundary the cell could still
  // be completely inside but a conservatice approach is chosen.
  // Since we only check for edges we guess it might intersect.
  //-> in this case we will generate a volume integral over whole cell!
  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<2>::reclassify(
    const TriangulationType &tria_unfitted)
  {
    quad_dg_face_vec.clear(); // move somewhere else!!
    location_to_geometry_vec.clear();
    location_to_geometry_vec.resize(tria_unfitted.n_active_cells());

    CGAL::Bounded_side inside_domain;
    if (boolean_operation == BooleanOperation::compute_intersection)
      {
        inside_domain = CGAL::ON_BOUNDED_SIDE;
      }
    else if (boolean_operation == BooleanOperation::compute_difference)
      {
        inside_domain = CGAL::ON_UNBOUNDED_SIDE;
      }
    else
      {
        DEAL_II_ASSERT_UNREACHABLE();
      }

    // now find out if inside or not
    for (const auto &cell : tria_unfitted.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        CGALPolygon polygon_cell;
        dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

        // requires smooth boundaries otherwise it might make mistakes
        // since it only checks for edge nodes
        // Assert checks for this in debug mode but this is very expensive
        unsigned int inside_count = 0;
        for (size_t i = 0; i < polygon_cell.size(); ++i)
          {
            const auto &p_1 = *(polygon_cell.begin() + i);
            const auto &p_2 =
              *(polygon_cell.begin() + ((i + 1) % polygon_cell.size()));
            const CGALPoint2 mid_point(0.5 * (p_1[0] + p_2[0]),
                                       0.5 * (p_1[1] + p_2[1]));

            auto result_1 = CGAL::bounded_side_2(fitted_2D_mesh.begin(),
                                                 fitted_2D_mesh.end(),
                                                 p_1);
            inside_count += (result_1 == inside_domain);

            auto result_2 = CGAL::bounded_side_2(fitted_2D_mesh.begin(),
                                                 fitted_2D_mesh.end(),
                                                 mid_point);
            inside_count += (result_2 == inside_domain);
          }
        if (inside_count == 0) // case 1: all vertices outside or on boundary:
                               // not considered (outside)
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::outside;
            // Assert(!CGAL::do_intersect(fitted_2D_mesh, polygon_cell),
            // ExcMessage("cell classified as outside although intersected"));
          }
        else if (inside_count ==
                 2 * cell->n_vertices()) // case 2: all vertices inside:
                                         // considered (inside)
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::inside;
          }
        else // case 3: at least one vertex inside and at least one on boundary
             // or outside
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::intersected;
          }
        // Note: construction of case 1 and 3 make sure that if two vertices are
        // on the boundary the cell
        // inside is considered cut and takes account for the face integral. The
        // cell outside also has vertices on the boundary but is ignored
        // because then the boundary integral would be performed twice
      }
  }

  template <>
  template <typename TriangulationType>
  void
  GridGridIntersectionQuadratureGenerator<3>::reclassify(
    const TriangulationType &tria_unfitted)
  {
    location_to_geometry_vec.clear();
    location_to_geometry_vec.resize(tria_unfitted.n_active_cells());

    CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K> inside_test(
      fitted_surface_mesh);

    CGAL::Bounded_side inside_domain;
    if (boolean_operation == BooleanOperation::compute_intersection)
      {
        inside_domain = CGAL::ON_BOUNDED_SIDE;
      }
    else if (boolean_operation == BooleanOperation::compute_difference)
      {
        inside_domain = CGAL::ON_UNBOUNDED_SIDE;
      }
    else
      {
        DEAL_II_ASSERT_UNREACHABLE();
      }

    for (const auto &cell : tria_unfitted.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        unsigned int inside_count = 0;
        for (size_t i = 0; i < cell->n_vertices(); i++)
          {
            auto result = inside_test(
              dealii_point_to_cgal_point<CGALPoint, 3>(cell->vertex(i)));
            inside_count += (result == inside_domain);
          }

        for (size_t i = 0; i < cell->n_faces(); i++)
          {
            auto result = inside_test(dealii_point_to_cgal_point<CGALPoint, 3>(
              cell->face(i)->center()));
            inside_count += (result == inside_domain);
          }

        if (inside_count == 0)
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::outside;
          }
        else if (inside_count == cell->n_vertices() + cell->n_faces())
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::inside;
          }
        else
          {
            location_to_geometry_vec[cell->active_cell_index()] =
              NonMatching::LocationToLevelSet::intersected;
          }
      }
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::generate(
    const typename Triangulation<2>::cell_iterator &cell)
  {
    // generate polygon for current cell
    CGALPolygon polygon_cell;
    dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

    // perform boolean operation on cell and fitted mesh
    // result is a polygon with holes
    std::vector<CGALPolygonWithHoles> polygon_out_vec;
    compute_boolean_operation(polygon_cell,
                              fitted_2D_mesh,
                              boolean_operation,
                              polygon_out_vec);

    // quadrature area in a cell could be split into two polygons
    // occurrence is not expected for smooth boundaries
    // -> outer for loop here for eventual extension
    Assert(polygon_out_vec.size() == 1,
           ExcMessage(
             "Not a single polygon with holes, disconnected domain!!"));

    std::vector<std::array<dealii::Point<2>, 3>> vec_of_simplices;
    for (const auto &polygon_out : polygon_out_vec)
      {
        Assert(polygon_out.outer_boundary().is_simple(),
               ExcMessage("The Polygon outer boundary is not simple"));
        // quadrature area in a cell cannot be a polygon with holes
        Assert(!polygon_out.has_holes(), ExcMessage("The Polygon has holes"));

        // partition polygon into convex polygons
        // these can be meshed as convex hull
        // Note: could use CGAL::approx_convex_partition_2
        Traits                       partition_traits;
        std::list<Traits::Polygon_2> convex_polygons;
        CGAL::optimal_convex_partition_2(
          polygon_out.outer_boundary().vertices_begin(),
          polygon_out.outer_boundary().vertices_end(),
          std::back_inserter(convex_polygons),
          partition_traits);

        Triangulation2 tria;
        for (const auto &convex_poly : convex_polygons)
          {
            tria.insert(convex_poly.vertices_begin(),
                        convex_poly.vertices_end());

            tria.insert_constraint(convex_poly.vertices_begin(),
                                   convex_poly.vertices_end(),
                                   true);

            // Extract simplices and construct quadratures
            for (const auto &face : tria.finite_face_handles())
              {
                std::array<dealii::Point<2>, 3> simplex;
                std::array<dealii::Point<2>, 3> unit_simplex;
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    simplex[i] =
                      cgal_point_to_dealii_point<2>(face->vertex(i)->point());
                  }
                mapping->transform_points_real_to_unit_cell(cell,
                                                            simplex,
                                                            unit_simplex);
                vec_of_simplices.push_back(unit_simplex);
              }

            tria.clear();
          }
      }
    quad_cells =
      QGaussSimplex<2>(quadrature_order).mapped_quadrature(vec_of_simplices);

    // surface quadrature
    std::vector<Quadrature<1>> quadrature_dg_faces_cell;
    quadrature_dg_faces_cell.resize(cell->n_faces());

    std::vector<Point<2>>     quadrature_points;
    std::vector<double>       quadrature_weights;
    std::vector<Tensor<1, 2>> normals;
    for (const auto &polygon_out : polygon_out_vec)
      {
        for (const auto &edge_cut : polygon_out.outer_boundary().edges())
          {
            unsigned int dg_face_index = cell->n_faces() + 1;
            auto         p_cut_1       = edge_cut.source();
            auto         p_cut_2       = edge_cut.target();
            for (const unsigned int i : cell->face_indices())
              {
                const typename Triangulation<2, 2>::face_iterator &face =
                  cell->face(i);
                if (face->at_boundary() ||
                    location_to_geometry(cell->neighbor(i)) ==
                      NonMatching::LocationToLevelSet::outside)
                  {
                    continue;
                  }

                auto p_uncut_1 =
                  dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(0));
                auto p_uncut_2 =
                  dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(1));

                if (CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_1) &&
                    CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_2))
                  {
                    dg_face_index = i;
                    break;
                  }
              }

            if (dg_face_index == cell->n_faces() + 1)
              {
                std::array<dealii::Point<2>, 2> unit_segment;
                mapping->transform_points_real_to_unit_cell(
                  cell,
                  {{cgal_point_to_dealii_point<2>(p_cut_1),
                    cgal_point_to_dealii_point<2>(p_cut_2)}},
                  unit_segment);

                auto quadrature =
                  QGaussSimplex<1>(quadrature_order)
                    .compute_affine_transformation(unit_segment);
                auto points  = quadrature.get_points();
                auto weights = quadrature.get_weights();

                // compute normals
                Tensor<1, 2> vector = unit_segment[1] - unit_segment[0];
                Tensor<1, 2> normal;
                if (boolean_operation == BooleanOperation::compute_intersection)
                  {
                    normal[0] = vector[1];
                    normal[1] = -vector[0];
                  }
                else if (boolean_operation ==
                         BooleanOperation::compute_difference)
                  {
                    normal[0] = -vector[1];
                    normal[1] = vector[0];
                  }
                else
                  {
                    DEAL_II_ASSERT_UNREACHABLE();
                  }

                normal /= normal.norm();


                quadrature_points.insert(quadrature_points.end(),
                                         points.begin(),
                                         points.end());
                quadrature_weights.insert(quadrature_weights.end(),
                                          weights.begin(),
                                          weights.end());
                normals.insert(normals.end(), quadrature.size(), normal);
              }
            else if (false) // change to true if want to use precomputed dg
              {
                // auto p_unit_1 =
                // mapping->project_real_point_to_unit_point_on_face( cell,
                // face_index, cgal_point_to_dealii_point<2>(p_cut_1));

                // auto p_unit_2 =
                // mapping->project_real_point_to_unit_point_on_face( cell,
                // face_index, cgal_point_to_dealii_point<2>(p_cut_2));

                // Quadrature<1> quadrature = QGaussSimplex<1>(quadrature_order)
                //                        .compute_affine_transformation({{p_unit_1,
                //                                                       p_unit_2}});

                // quadrature_dg_faces_cell[dg_face_index] = quadrature;
              }
          }
      }
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<2>(quadrature_points,
                                                             quadrature_weights,
                                                             normals);
    // quad_dg_face_vec[cell->active_cell_index()] = quadrature_dg_faces_cell;
    // //if want to use precomputed dg
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::generate(
    const typename Triangulation<3>::cell_iterator &cell)
  {
    CGAL::Surface_mesh<CGALPoint> surface_cell;
    dealii_cell_to_cgal_surface_mesh(cell, *mapping, surface_cell);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_cell);

    CGAL::Surface_mesh<CGALPoint> out_surface;
    compute_boolean_operation(surface_cell,
                              fitted_surface_mesh,
                              boolean_operation,
                              out_surface);

    // Fill triangulation with vertices from surface mesh
    CGALTriangulation tria;
    tria.insert(out_surface.points().begin(), out_surface.points().end());

    CGAL::Side_of_triangle_mesh<CGAL::Surface_mesh<CGALPoint>, K> inside_test(
      out_surface);

    // Extract simplices and construct quadratures
    std::vector<std::array<dealii::Point<3>, 4>> vec_of_simplices;
    for (const auto &face : tria.finite_cell_handles())
      {
        std::array<CGALPoint, 4>        simplex_cgal;
        std::array<dealii::Point<3>, 4> simplex;
        std::array<dealii::Point<3>, 4> unit_simplex;
        for (unsigned int i = 0; i < 4; ++i)
          {
            simplex_cgal[i] = face->vertex(i)->point();
            simplex[i] =
              cgal_point_to_dealii_point<3>(face->vertex(i)->point());
          }

        auto centroid = CGAL::centroid(simplex_cgal[0],
                                       simplex_cgal[1],
                                       simplex_cgal[2],
                                       simplex_cgal[3]);
        // dont consider convex hull triangles that are not inside
        if (inside_test(centroid) != CGAL::ON_BOUNDED_SIDE)
          continue;

        mapping->transform_points_real_to_unit_cell(cell,
                                                    simplex,
                                                    unit_simplex);
        vec_of_simplices.push_back(unit_simplex);
      }
    quad_cells =
      QGaussSimplex<3>(quadrature_order).mapped_quadrature(vec_of_simplices);

    // surface quadrature
    std::vector<Point<3>>     quadrature_points;
    std::vector<double>       quadrature_weights;
    std::vector<Tensor<1, 3>> normals;
    double ref_area = std::pow(cell->minimum_vertex_distance(), 2) * 0.0000001;
    for (const auto &out_surface_face : out_surface.faces())
      {
        if (CGAL::abs(CGAL::Polygon_mesh_processing::face_area(out_surface_face,
                                                               out_surface)) <
            ref_area)
          {
            continue;
          }

        unsigned int             dg_face_index = cell->n_faces() + 1;
        std::array<CGALPoint, 3> simplex;
        int                      i = 0;
        for (const auto &vertex :
             CGAL::vertices_around_face(out_surface.halfedge(out_surface_face),
                                        out_surface))
          {
            simplex[i] = out_surface.point(vertex);
            i += 1;
          }

        for (const unsigned int i_dealii_face : cell->face_indices())
          {
            const typename Triangulation<3, 3>::face_iterator &face =
              cell->face(i_dealii_face);
            if (face->at_boundary() ||
                location_to_geometry(cell->neighbor(i_dealii_face)) ==
                  NonMatching::LocationToLevelSet::outside)
              {
                continue;
              }

            int count = 0;
            for (unsigned int i_dealii_vertex = 0;
                 i_dealii_vertex < face->n_vertices();
                 ++i_dealii_vertex)
              {
                auto cgal_point = dealii_point_to_cgal_point<CGALPoint, 3>(
                  face->vertex(i_dealii_vertex));
                count += CGAL::coplanar(simplex[0],
                                        simplex[1],
                                        simplex[2],
                                        cgal_point);
              }

            if (count >= 3)
              {
                dg_face_index = i_dealii_face;
                break;
              }
          }

        if (dg_face_index == cell->n_faces() + 1)
          {
            std::array<Point<3>, 3> unit_simplex;

            // compute quadrature and fill vectors
            mapping->transform_points_real_to_unit_cell(
              cell,
              {{cgal_point_to_dealii_point<3>(simplex[0]),
                cgal_point_to_dealii_point<3>(simplex[1]),
                cgal_point_to_dealii_point<3>(simplex[2])}},
              unit_simplex);
            auto quadrature = QGaussSimplex<2>(quadrature_order)
                                .compute_affine_transformation(unit_simplex);
            auto points  = quadrature.get_points();
            auto weights = quadrature.get_weights();
            quadrature_points.insert(quadrature_points.end(),
                                     points.begin(),
                                     points.end());
            quadrature_weights.insert(quadrature_weights.end(),
                                      weights.begin(),
                                      weights.end());

            const Tensor<1, 3> v1 = cgal_point_to_dealii_point<3>(simplex[2]) -
                                    cgal_point_to_dealii_point<3>(simplex[1]);
            const Tensor<1, 3> v2 = cgal_point_to_dealii_point<3>(simplex[0]) -
                                    cgal_point_to_dealii_point<3>(simplex[1]);
            Tensor<1, 3> normal = cross_product_3d(v1, v2);
            normal /= normal.norm();
            normals.insert(normals.end(), quadrature.size(), normal);
          }
      }
    quad_surface = NonMatching::ImmersedSurfaceQuadrature<3>(quadrature_points,
                                                             quadrature_weights,
                                                             normals);
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<2>::generate_dg_face(
    const typename Triangulation<2>::cell_iterator &cell,
    unsigned int                                    face_index)
  {
    const typename Triangulation<2, 2>::face_iterator &face =
      cell->face(face_index);
    if (face->at_boundary() ||
        location_to_geometry(cell->neighbor(face_index)) ==
          NonMatching::LocationToLevelSet::outside)
      {
        quad_dg_face = Quadrature<1>();
        return;
      }

    CGALPolygon polygon_cell;
    dealii_cell_to_cgal_polygon(cell, *mapping, polygon_cell);

    std::vector<CGALPolygonWithHoles> polygon_out_vec;

    compute_boolean_operation(polygon_cell,
                              fitted_2D_mesh,
                              boolean_operation,
                              polygon_out_vec);

    std::vector<Point<1>> quadrature_points;
    std::vector<double>   quadrature_weights;
    for (const auto &polygon_out : polygon_out_vec)
      {
        for (const auto &edge_cut : polygon_out.outer_boundary().edges())
          {
            bool dg_face = false;
            auto p_cut_1 = edge_cut.source();
            auto p_cut_2 = edge_cut.target();

            auto p_uncut_1 =
              dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(0));
            auto p_uncut_2 =
              dealii_point_to_cgal_point<CGALPoint2, 2>(face->vertex(1));
            if (CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_1) &&
                CGAL::collinear(p_uncut_1, p_uncut_2, p_cut_2))
              {
                dg_face = true;
              }

            if (dg_face)
              {
                auto p_unit_1 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell, face_index, cgal_point_to_dealii_point<2>(p_cut_1));

                auto p_unit_2 =
                  mapping->project_real_point_to_unit_point_on_face(
                    cell, face_index, cgal_point_to_dealii_point<2>(p_cut_2));

                Quadrature<1> quadrature =
                  QGaussSimplex<1>(quadrature_order)
                    .compute_affine_transformation({{p_unit_1, p_unit_2}});

                auto points  = quadrature.get_points();
                auto weights = quadrature.get_weights();
                quadrature_points.insert(quadrature_points.end(),
                                         points.begin(),
                                         points.end());
                quadrature_weights.insert(quadrature_weights.end(),
                                          weights.begin(),
                                          weights.end());
              }
          }
      }
    quad_dg_face = Quadrature<1>(quadrature_points, quadrature_weights);
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::generate_dg_face(
    const typename Triangulation<3>::cell_iterator &cell,
    unsigned int                                    face_index)
  {
    const typename Triangulation<3, 3>::face_iterator &face =
      cell->face(face_index);
    if (face->at_boundary() ||
        location_to_geometry(cell->neighbor(face_index)) ==
          NonMatching::LocationToLevelSet::outside)
      {
        quad_dg_face = Quadrature<2>();
        return;
      }

    CGAL::Surface_mesh<CGALPoint> surface_cell;
    dealii_cell_to_cgal_surface_mesh(cell, *mapping, surface_cell);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_cell);

    CGAL::Surface_mesh<CGALPoint> out_surface;
    compute_boolean_operation(surface_cell,
                              fitted_surface_mesh,
                              boolean_operation,
                              out_surface);

    std::vector<CGALPoint> dealii_vertices;
    for (unsigned int i_dealii_vertex = 0; i_dealii_vertex < face->n_vertices();
         ++i_dealii_vertex)
      {
        dealii_vertices.push_back(dealii_point_to_cgal_point<CGALPoint, 3>(
          face->vertex(i_dealii_vertex)));
      }

    std::vector<Point<2>> quadrature_points;
    std::vector<double>   quadrature_weights;
    for (const auto &face_surface : out_surface.faces())
      {
        int                      i_cgal_vertex = 0;
        std::array<CGALPoint, 3> cgal_vertices;
        for (const auto &vertex :
             CGAL::vertices_around_face(out_surface.halfedge(face_surface),
                                        out_surface))
          {
            cgal_vertices[i_cgal_vertex] = out_surface.point(vertex);
            i_cgal_vertex += 1;
          }

        int count = 0;
        for (auto &vertex : dealii_vertices)
          {
            count += CGAL::coplanar(cgal_vertices[0],
                                    cgal_vertices[1],
                                    cgal_vertices[2],
                                    vertex);
          }

        if (count >= 3)
          {
            std::array<Point<2>, 3> simplex;
            int                     i = 0;
            for (const auto &vertex :
                 CGAL::vertices_around_face(out_surface.halfedge(face_surface),
                                            out_surface))
              {
                auto dealii_point =
                  cgal_point_to_dealii_point<3>(out_surface.point(vertex));
                simplex[i] = mapping->project_real_point_to_unit_point_on_face(
                  cell, face_index, dealii_point);
                i += 1;
              }

            auto quadrature = QGaussSimplex<2>(quadrature_order)
                                .compute_affine_transformation(simplex);

            auto points  = quadrature.get_points();
            auto weights = quadrature.get_weights();
            quadrature_points.insert(quadrature_points.end(),
                                     points.begin(),
                                     points.end());
            quadrature_weights.insert(quadrature_weights.end(),
                                      weights.begin(),
                                      weights.end());
          }
      }
    quad_dg_face = Quadrature<2>(quadrature_points, quadrature_weights);
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
  GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature_dg_face()
    const
  {
    return quad_dg_face;
  }

  template <int dim>
  Quadrature<dim - 1>
  GridGridIntersectionQuadratureGenerator<dim>::get_inside_quadrature_dg_face(
    const typename Triangulation<dim>::cell_iterator &cell,
    unsigned int                                      face_index) const
  {
    auto it = quad_dg_face_vec.find(cell->active_cell_index());
    if (it == quad_dg_face_vec.end())
      {
        return Quadrature<dim - 1>();
      }
    else
      {
        return it->second[face_index];
      }
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
  void
  GridGridIntersectionQuadratureGenerator<2>::output_fitted_mesh() const
  {
    std::string   filename = "fitted_polygon.vtu";
    std::ofstream file(filename);
    if (!file)
      {
        std::cerr << "Error opening file for writing: " << filename
                  << std::endl;
        return;
      }

    const std::size_t n = fitted_2D_mesh.size();

    file << R"(<?xml version="1.0"?>)"
         << "\n";
    file
      << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)"
      << "\n";
    file << R"(  <UnstructuredGrid>)"
         << "\n";
    file << R"(    <Piece NumberOfPoints=")" << n << R"(" NumberOfCells="1">)"
         << "\n";

    // Points section
    file << R"(      <Points>)"
         << "\n";
    file
      << R"(        <DataArray type="Float64" NumberOfComponents="3" format="ascii">)"
      << "\n";

    for (const auto &p : fitted_2D_mesh.container())
      {
        file << p.x() << " " << p.y() << " 0 ";
      }
    file << "\n";

    file << R"(        </DataArray>)"
         << "\n";
    file << R"(      </Points>)"
         << "\n";

    // Cells section
    // Connectivity: indices of vertices in order
    file << R"(      <Cells>)"
         << "\n";

    // Connectivity
    file
      << R"(        <DataArray type="Int32" Name="connectivity" format="ascii">)";
    for (std::size_t i = 0; i < n; ++i)
      {
        file << i << " ";
      }
    file << R"(</DataArray>)"
         << "\n";

    // Offsets: cumulative count of vertices after each cell
    // Here only one cell with n vertices
    file << R"(        <DataArray type="Int32" Name="offsets" format="ascii">)";
    file << n << R"(</DataArray>)"
         << "\n";

    // Types: VTK cell type for polygon is 7
    // (See https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
    file
      << R"(        <DataArray type="UInt8" Name="types" format="ascii">7</DataArray>)"
      << "\n";

    file << R"(      </Cells>)"
         << "\n";

    file << R"(    </Piece>)"
         << "\n";
    file << R"(  </UnstructuredGrid>)"
         << "\n";
    file << R"(</VTKFile>)"
         << "\n";

    file.close();
  }

  template <>
  void
  GridGridIntersectionQuadratureGenerator<3>::output_fitted_mesh() const
  {
    CGAL::IO::write_polygon_mesh("fitted_surface_mesh.stl",
                                 fitted_surface_mesh);
  }



// Explicit instantiations.
//
// We don't build the instantiations.inst file if deal.II isn't
// configured with CGAL, but doxygen doesn't know that and tries to
// find that file anyway for parsing -- which then of course it fails
// on. So exclude the following from doxygen consideration.
#  ifndef DOXYGEN
#    include "cgal/grid_grid_intersection_quadrature_generator.inst"
#  endif

} // namespace  CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
