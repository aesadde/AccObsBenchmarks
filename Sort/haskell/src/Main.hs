{-# LANGUAGE BangPatterns, ScopedTypeVariables#-}
module Main where

import RadixSortAcc as A
import SortNetObs

import Criterion.Main
import Criterion.Types
import Prelude                                  as P
import Text.Printf
import System.Environment
import qualified Data.Vector.Storable as V
import Random
import Data.Array.Accelerate.CUDA as A
import Data.Array.Accelerate      as A
import System.Random.MWC
import qualified Foreign.CUDA.Driver                    as CUDA

import Data.List as L (sort)
import Obsidian.Run.CUDA.Exec

benchmarks xs_arr ctx kern inputs size sorted =
    defaultMainWith benchConfig
      [ bgroup ("Sorting 2^" P.++ (show size) P.++ " elements")
        [
        bench ("accelerate") $ whnf (A.run1 A.sort) xs_arr,
        bench ("obsidian")   $ whnfIO (runSortObs ctx kern inputs size sorted)
        ]
      ]
      --
-- This configuration enables garbage collection between benchmarks. It is a
-- good idea to do so. Otherwise GC might distort your results
benchConfig :: Config
benchConfig = defaultConfig { forceGC = True}

main :: IO ()
main = do
    args <- getArgs
    let inSize = read (args P.!! 0) :: Int
    let size = 2^inSize

    -- Obsidian configuration
    ctx <- initialise
    !kern <- captureIO "sort_kernel" (props ctx) 32 sortObs
    !(inputs' :: V.Vector Word32) <- mkRandomVec size
    let inputs = V.map (`mod` 64) inputs'
    let sorted = L.sort (V.toList inputs)

    -- Accelerate configuration
    !xs_arr <- randomArrayIO (const uniform) (Z :. size):: IO (Vector Int32)

    withArgs (P.tail args) $ benchmarks xs_arr ctx kern inputs size sorted
