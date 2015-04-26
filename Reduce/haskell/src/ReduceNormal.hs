{-# LANGUAGE ScopedTypeVariables, BangPatterns, TypeOperators #-}
module ReduceNormal where

import Data.Array.Accelerate as A hiding ((++),lift,fromIntegral,sum)
import ReduceObsidian
import ReduceAccelerate
import Prelude
import qualified Prelude as P hiding (zipWith,sum,replicate,take,drop,iterate)
import Control.Monad.State
import qualified Data.Vector.Storable as V
import Control.DeepSeq
import System.Environment
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Obsidian.Run.CUDA.Exec
import Obsidian(splitUp)

-- | performAcc executes and measures the time of parallel reduce in Accelerate
performAcc :: (Elt a, Shape ix, IsNum a) => A.Array (ix :. Int) a -> IO ()
performAcc input = do
          s <- getCurrentTime
          print $ accReduce (A.use input)
          e2 <- getCurrentTime
          print $ "Accelerate " ++ show (diffUTCTime e2 s)

-- | performObs executes and measures the time of parallel reduce in Obsidian
performObs threads elts inputV result elsPT =
  withCUDA $
  do
    let tdsPerBlock = threads * elsPT
    let blcks = (elts `div` tdsPerBlock) :: Int
    let size = fromIntegral (elts `div` blcks)

    -- compile the CUDA binary
    startC <- lift getCurrentTime
    ker <- capture (P.fromIntegral threads) (reduceKernel (+) . splitUp size)
    endC <- lift getCurrentTime

    -- allocate memory and run the kernel
    sa <- lift getCurrentTime
    useVector inputV $ \i ->
      allocaVector (blcks) $ \(o :: CUDAVector Double) ->
        do
          start <- lift getCurrentTime
          o <== (fromIntegral blcks, ker) <> i
          r <- copyOut o
          let gpuR = sum (V.toList r)
          lift $ print gpuR
          end <- lift getCurrentTime
          lift $ print $ "(" ++ show (diffUTCTime end startC) -- complete
           ++ ";"
           ++ show (diffUTCTime endC startC) -- compilation
           ++ ";"
           ++ show (diffUTCTime end sa) -- kernel w transfers
           ++ ";"
           ++ show (diffUTCTime end start) ++ ")" -- kernel no transfer
          lift $ print $ "Obsidian R " ++ show (gpuR == result)

-- |runNormal: Configure all options and run both reduce programs
runNormal :: IO ()
runNormal = do
        args <- getArgs
        let s = read (args P.!! 0) :: Int
        let elsPT = read (args P.!! 1) :: Int
        let threads = 1024
        let size = 3200*1024*s
        let sw = (P.fromIntegral size) :: Word32
        let !inputAcc = (A.fromList (A.Z A.:. size) [1..(P.fromIntegral size)]) :: (Vector Double)
        let tdsPerBlock = threads * elsPT
        let blcks = size `div` tdsPerBlock :: Int
        let blockSize = P.fromIntegral(size `div` blcks)
        let (inputV  :: V.Vector Double) = V.fromList [1..(P.fromIntegral size)]
        let result = V.sum inputV

        startC <- getCurrentTime
        _ <- result `deepseq` return()
        print result
        endC <- getCurrentTime
        print $ "Sequential time" ++ show (diffUTCTime endC startC)

        print $ "(Elems " ++ show size ++ ")"
        performAcc inputAcc
        performObs threads size inputV result elsPT
