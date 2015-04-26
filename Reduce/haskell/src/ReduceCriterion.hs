{-# LANGUAGE NoMonomorphismRestriction,
             ScopedTypeVariables, BangPatterns#-}
module ReduceCriterion where

import Criterion.Main
import Criterion.Types
import Data.Array.Accelerate as A hiding ((++),fromIntegral,lift)
import Data.Array.Accelerate.CUDA as A

import ReduceObsidian
import ReduceAccelerate

import System.Environment
import Prelude
import qualified Prelude as P hiding (zipWith,replicate,take,drop,iterate)
import Control.Monad.State

import qualified Data.Vector.Storable as V
import Obsidian.Run.CUDA.Exec
import Obsidian
import Control.DeepSeq(deepseq)
import Data.Word

-- |benchmarks: Uses criterion to measure the performance of Accelerate,
-- Obsidian and a sequential version directly written in Haskell
benchmarks inputAcc tdsPerBlock size inputV result kern ctx =
        defaultMainWith benchConfig
           [ bgroup ("Elems " ++ show size)
            [
              bench "Accelerate" $ whnf accReduce (A.use inputAcc),
              bench "Obsidian"   $ whnfIO (performObsidian ctx kern tdsPerBlock size inputV result),
              bench "Sequential" $ whnf V.sum inputV
            ]
          ]

-- |performObsidian: runs the given CUDA binary on the GPU
performObsidian ctx kern tdsPerBlock elts inputV result =
  withCUDA' ctx $
  do
    let blcks = (elts `div` tdsPerBlock) :: Int
    let size = fromIntegral (elts `div` blcks)

    useVector inputV $ \i ->
      allocaVector (blcks) $ \(o :: CUDAVector Double) ->
        do
          o <== (fromIntegral blcks, kern) <> i
          r <- copyOut o
          let gpuR = P.sum (V.toList r)
          return $ "Obsidian R " ++ show (gpuR == result)

-- This configuration enables garbage collection between benchmarks. It is a
-- good idea to do so. Otherwise GC might distort your results
benchConfig :: Config
benchConfig = defaultConfig { forceGC = True}

runCriterion = do
    -- use first arg to calculate size of array
    args <- getArgs
    let s = read (args P.!! 0 ) :: Int

    -- Obsidian config
    let threads = 1024
    let size = 3200*1024*s
    let sw = (P.fromIntegral size) :: Word32

    let tdsPerBlock = threads * 32
    let blcks = size `div` tdsPerBlock :: Int
    let blockSize = P.fromIntegral(size `div` blcks)

    -- Obsidian compilation before benchmarks
    ctx <- initialise
    kern <- captureIO "kernel" (props ctx) (P.fromIntegral threads) (reduceKernel (+) . splitUp blockSize)

    -- Accelerate and Seq config
    let inputAcc = (A.fromList (A.Z A.:. size) [1..(P.fromIntegral size)]) :: (A.Array DIM1 Double)
    let (inputV  :: V.Vector Double) = V.fromList [1..(P.fromIntegral size)]
    let result = V.sum inputV
    _ <- result `deepseq` return()

    -- use remaining arguments with criterion
    withArgs (P.tail args) $ benchmarks inputAcc tdsPerBlock size inputV result kern ctx
