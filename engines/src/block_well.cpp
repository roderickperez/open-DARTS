#include "block_well.h"
block_well::block_well ()
{
  well_block_volume = 0.07; // 1 m high, 0.3 m diameter
  well_block_trans = 100000; 
  current_rates = 0;
  name = "";
}

