{-# LANGUAGE FlexibleContexts, TypeOperators, GADTs, BangPatterns#-}

module MatMulOpt(runMatMulObs, matMulSh) where
import Obsidian
import Obsidian.Run.CUDA.Exec as CUDA
import qualified Data.Vector.Storable as V

import Control.Monad.State
import Prelude as P hiding (zipWith)
import Data.Word




dotProd :: (Num a, Data a) => Pull Word32 a -> Pull Word32 a -> Push Thread Word32 a
dotProd a b = execThread' $ seqReduce (+) (zipWith (*) a b)

transpose :: Pull Word32 (Pull Word32 a) -> Pull Word32 (Pull Word32 a)
transpose arr = mkPull n1 (\i -> mkPull n2 (\j -> (arr ! j) ! i))
  where
    n2 = len arr
    n1 = len (arr ! 0)

runMatMulObs ctx kern size side =
  withCUDA' ctx $
  do
    useVector (V.replicate size (2.0 :: Float)) $ \i1 ->
      useVector (V.replicate size (2.0 :: Float)) $ \i2 ->
        allocaVector size $ \ o ->
      do
        o <== (fromIntegral size,kern) <> i1 <> i2
        r <- copyOut o
        lift $ print (V.toList r)
