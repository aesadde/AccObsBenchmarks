{-# LANGUAGE FlexibleContexts, TypeOperators, GADTs #-}
module MatMul (matMul,matMulSh,runMatMulObs) where
import Obsidian
import Obsidian.Run.CUDA.Exec as CUDA

import Prelude hiding (zipWith)
import Data.Word
import Control.Monad.State
import qualified Data.Vector.Storable as V

-- |matMul: Main function. Spreads the multiplication on the GPU in parallel
matMul :: (Num a, Data a) => Pull Word32 (Pull Word32 a) -> Pull Word32 (Pull Word32 a) -> Push Grid Word32 a
matMul a b = asGridMap body a
    where
        body x = matMulRow x (transpose b)

-- |matMulSh: Matrix multiplication using shared-memory (matMulRow')
matMulSh :: (ASize l, Data b, Num b) => Pull l (Pull Word32 b) -> Pull Word32 (Pull Word32 b) -> Push Grid l b
matMulSh a b = asGridMap body a
  where
    body x = matMulRow' x (transpose b)

-- | matMulRow: Original Obsidian implementation.
matMulRow :: (Num a, Data a) => Pull Word32 a -> Pull Word32 (Pull Word32 a) -> Push Block Word32 a
matMulRow row mat = asBlockMap (dotProd row) mat

-- | matMulRow': Uses shared memory in an attempt to optimise the normal version
matMulRow' :: (Num a, Data a) => Pull Word32 a -> SPull (Pull Word32 a) -> Push Block Word32 a
matMulRow' row mat = Obsidian.exec $ do
    row' <- compute $ push row
    return $ asBlockMap (dotProd row') mat

-- | dotProd: Performs dot-product at the Thread level
dotProd :: (Num a, Data a) => Pull Word32 a -> Pull Word32 a -> Push Thread Word32 a
dotProd a b = execThread' $ seqReduce (+) (zipWith (*) a b)

-- | transpose: transposes the matrix
transpose :: Pull Word32 (Pull Word32 a) -> Pull Word32 (Pull Word32 a)
transpose arr = mkPull n1 (\i -> mkPull n2 (\j -> (arr ! j) ! i))
    where
        n2 = len arr
        n1 = len (arr ! 0)


-- | runMatMulObs: runs the matrix multiplication on the GPU with the given CUDA binary
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
