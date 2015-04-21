#include "dolfin.h"
#include <sys/types.h>
#include <unistd.h>
#include <fstream>

namespace dolfin
{
  std::size_t getMemoryUsage(bool rss=true)
  {
    // Get process ID and page size
    const std::size_t pid = getpid();
    const std::size_t page_size = getpagesize();

    // Prepare statm file
    std::stringstream filename;
    filename << "/proc/" << pid << "/statm";
    std::ifstream statm;

    // Read number of pages from statm
    statm.open(filename.str().c_str());
    if (!statm)
      std::cout << "Unable to open statm file for process." << std::endl;
    std::size_t num_pages;
    statm >> num_pages;
    if (rss)
      statm >> num_pages;
    statm.close();

    // Convert to MB and report memory usage
    const std::size_t num_kb = num_pages*page_size / 1024;
    const std::size_t num_mb = num_pages*page_size / (1024*1024);
//     if (verbose)
//       std::cout << "Memory usage: " << num_mb << " MB" << std::endl;
    return num_mb;
  }
  
  void SetMatrixValue(GenericMatrix& A, double val)
  {
    std::vector<std::size_t> columns;
    std::vector<double> values;
    std::vector<std::vector<std::size_t> > allcolumns;
    std::vector<std::vector<double> > allvalues;
    
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    const std::size_t m = row_range.second - row_range.first;
    std::cout << m << std::endl;
    for (std::size_t row = 0; row < m; row++)
    {
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      
      A.getrow(global_row, columns, values);
      for (std::size_t i = 0; i < values.size(); i++)
        values[i] = val;
      
      allvalues.push_back(values);
      allcolumns.push_back(columns);
    }
    
    for (std::size_t row = 0; row < m; row++)
    {       
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      
      A.setrow(global_row, allcolumns[row], allvalues[row]);
    }
    A.apply("insert"); 
    
  }  
}
