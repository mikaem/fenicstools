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
}
