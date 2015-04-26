{-# LANGUAGE ScopedTypeVariables #-}
module ReduceAccelerate where

import Data.Array.Accelerate as A hiding ((++))
import Data.Array.Accelerate.CUDA (run)
import Data.Time.Clock (diffUTCTime, getCurrentTime)

accReduce arr = run $ fold (+) 0 arr


