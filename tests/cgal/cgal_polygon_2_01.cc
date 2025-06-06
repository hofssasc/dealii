// Convert a deal.II reference cell to a cgal polygon_2.
#include <deal.II/base/config.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/cgal/polygon_2.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>



#include "../tests.h"

using namespace CGALWrappers;

using K             = CGAL::Exact_predicates_exact_constructions_kernel;
using CGALPoint2    = CGAL::Point_2<K>;
using CGALPolygon   = CGAL::Polygon_2<K>;

void
test_quadrilateral()
{
    using namespace ReferenceCells;
    ReferenceCell r_cell = Quadrilateral;
    Triangulation<2, 2>  tria;

    const auto mapping =
        r_cell.template get_default_mapping<2, 2>(1);

    GridGenerator::reference_cell(tria, r_cell);

    CGALPolygon poly;

    const auto cell = tria.begin_active();
    dealii_cell_to_cgal_polygon(cell, *mapping, poly);

    deallog << "Quadrilateral: " << std::endl;
    deallog << "Simple polygon: " << std::boolalpha << poly.is_simple()
            << std::endl;
    deallog << "Counterclockwise oriented polygon: " << std::boolalpha
            << poly.is_counterclockwise_oriented() << std::endl;
    deallog << "deal vertices: " << cell->n_vertices() << ", cgal vertices "
            << poly.size() << std::endl;
    deallog << "deal measure: " << GridTools::volume(tria) << ", cgal measure: "
            << poly.area() << std::endl;

    for(size_t i = 0; i < poly.size(); ++i)
    {
        deallog << "Polygon vertex " << i << ": " << poly[i] << std::endl;
    }
    deallog << std::endl;
}

void
test_triangle()
{
    using namespace ReferenceCells;
    ReferenceCell r_cell = Triangle;
    Triangulation<2, 2>  tria;

    const auto mapping =
        r_cell.template get_default_mapping<2, 2>(1);

    GridGenerator::reference_cell(tria, r_cell);

    CGALPolygon poly;

    const auto cell = tria.begin_active();
    dealii_cell_to_cgal_polygon(cell, *mapping, poly);

    deallog << "Triangle: " << std::endl;
    deallog << "Simple polygon: " << std::boolalpha << poly.is_simple()
            << std::endl;
    deallog << "Counterclockwise oriented polygon: " << std::boolalpha
            << poly.is_counterclockwise_oriented() << std::endl;
    deallog << "deal vertices: " << cell->n_vertices() << ", cgal vertices "
            << poly.size() << std::endl;        
    deallog << "deal measure: " << GridTools::volume(tria) << ", cgal measure: "
            << poly.area() << std::endl;
    
    for(size_t i = 0; i < poly.size(); ++i)
    {
        deallog << "Polygon vertex " << i << ": " << poly[i] << std::endl;
    }
}

int
main()
{
  initlog();
  test_quadrilateral();
  test_triangle();
}