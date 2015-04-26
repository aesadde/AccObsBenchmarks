{-# LANGUAGE NoMonomorphismRestriction, ScopedTypeVariables, BangPatterns #-}
module Main where

import MatMul
import MatMulAcc

import Criterion.Main
import Criterion.Types

import System.Environment
import Prelude
import qualified Prelude as P hiding (zipWith,sum,take,drop,iterate)
import Control.Monad.State

import Obsidian.Run.CUDA.Exec(captureIO,initialise,props)
import Obsidian(fromDyn,toDyn,splitUp,Pull,DPush,EWord32,Grid)
import Control.DeepSeq(deepseq)

import Data.Array.Accelerate as A hiding ((++))
import Data.Array.Accelerate.CUDA as A

import qualified Foreign.CUDA.Driver                    as CUDA
import Data.Word

benchmarks ctx kern1 kern2 size side ma mb =
        defaultMainWith benchConfig
           [ bgroup (show side ++ " x "++ show side)
            [
              -- bench "Accelerate"            $ whnf (accMatMul (A.use ma)) (A.use mb),
              bench "Obsidian"              $ whnfIO (runMatMulObs ctx kern1 size side),
              bench "Obsidian Shared-Mem"   $ whnfIO (runMatMulObs ctx kern2 size side)

            ]
          ]

-- This configuration enables garbage collection between benchmarks. It is a
-- good idea to do so. Otherwise GC might distort your results
benchConfig :: Config
benchConfig = defaultConfig { forceGC = True}

main = do
    -- user configuration
    args <- getArgs
    let s = read (args P.!! 0 ) :: Int

    let threads = 1024
    let side = 2^s
    let size = side * side
    let s' = (P.fromIntegral side) :: Word32

    -- Obsidian configuration
    ctx <- initialise
    !kern1 <- captureIO "kernel" (props ctx) 1024 (\a b -> toDyn (matMulSh (fromDyn s' (splitUp s' a)) (fromDyn s' (splitUp s' b))))
    !kern2 <- captureIO "kernel" (props ctx) 1024 (\a b -> toDyn (matMul (fromDyn s' (splitUp s' a)) (fromDyn s' (splitUp s' b))))

    -- Accelerate configuration
    let mb = (A.fromList (Z :. side :. side) (P.replicate size 2) :: Array DIM2 Float)
    let ma = (A.fromList (Z :. side :. side) (P.replicate size 2) :: Array DIM2 Float)

    -- use remaining arguments with criterion
    withArgs (P.tail args) $ benchmarks ctx kern1 kern2 size side ma mb
