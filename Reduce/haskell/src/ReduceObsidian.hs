{-# LANGUAGE NoMonomorphismRestriction,
             ScopedTypeVariables#-}
module ReduceObsidian(reduceKernel) where

import Obsidian
import Data.Word
import Prelude hiding (map,zipWith,sum,replicate,take,drop,iterate)
import Obsidian.Run.CUDA.Exec

blockRed :: Data a => Word32 -> (a -> a -> a) -> SPull a -> BProgram (SPush Block a)
blockRed cutoff f arr
  | len arr == cutoff = return $ push $ fold1 f arr
  | otherwise = do
      let (a1,a2) = halve arr
      arr' <- compute (zipWith f a1 a2)
      blockRed cutoff f arr'

coalesceRed f arr =
    do arr' <- compute $ asBlockMap (execThread' . seqReduce f) (coalesce 32 arr)
       blockRed 2 f arr'

reduceKernel f arr = asGridMap body arr
  where
    body arr = execBlock (coalesceRed f arr)
