{-# LANGUAGE NoMonomorphismRestriction, ScopedTypeVariables, BangPatterns #-}
module Main where

import MandelObs as O
import MandelAcc as A

import Criterion.Main
import Criterion.Types

import System.Environment
import Prelude as P
import qualified Prelude as P hiding (zipWith,sum,take,drop,iterate)

import Obsidian.Run.CUDA.Exec(captureIO,initialise,props)
import Obsidian(fromDyn,toDyn,splitUp,Pull,DPush,EWord32,Grid)

import Data.Array.Accelerate as A hiding ((++))
import Data.Array.Accelerate.CUDA as A

import qualified Foreign.CUDA.Driver                    as CUDA
import Data.Word

benchmarks ctx kern threads runMandelAcc view size =
        defaultMainWith benchConfig
           [ bgroup ("Mandel ")
            [
            bench "Accelerate"            $ whnf runMandelAcc view,
            bench "Obsidian"              $ whnfIO (runMandelObs ctx kern threads size)
            ]
          ]

-- This configuration enables garbage collection between benchmarks. It is a
-- good idea to do so. Otherwise GC might distort your results
benchConfig :: Config
benchConfig = defaultConfig { forceGC = True}

main = do
    user configuration
    args <- getArgs
    let wht = read (args P.!! 0 ) :: Int
    let depth = 512

    Obsidian configuration
    ctx <- initialise
    !kern <- captureIO "kernel" (props ctx) (P.fromIntegral wht)  (mandel (P.fromIntegral wht) (P.fromIntegral wht))

    Accelerate configuration
    let
        view :: (Float,Float,Float,Float)
        view      = (-2.23, -1.15, 0.83, 1.15)
        view'     = A.fromList A.Z [view]
        runCUDA  = A.run1 (mandelbrot wht wht wht)

    withArgs (P.tail args) $ benchmarks ctx kern wht runCUDA view' (wht*wht)
