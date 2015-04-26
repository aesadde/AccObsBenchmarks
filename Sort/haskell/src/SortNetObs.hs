{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

module SortNetObs where

import Obsidian
import Obsidian.Run.CUDA.Exec as CUDA hiding (exec)
import qualified Data.Vector.Storable as V

import Prelude hiding (zip, reverse )
import qualified Prelude as P
import Control.Monad.State(lift)

import Data.Word

riffle' :: (t *<=* Block, Data a, ASize s) => Pull s a -> Push t s (a,a)
riffle' = push . uncurry zip . halve

compareSwap :: (Scalar a, Ord a) => (Exp a,Exp a) -> (Exp a,Exp a)
compareSwap (a,b) = ifThenElse (b <* a) (b,a) (a,b)

shexRev' :: (Array (Push t), Compute t, Data a)
         => ((a,a) -> (a,a)) -> SPull a -> SPush t a
shexRev' cmp arr =
  let (arr1,arr2) = halve arr
      arr2' = reverse arr2
      arr' = (push arr1) `append` (push arr2')
  in
   exec $ do
     arr'' <- compute arr'
     rep (logBaseI 2 (len arr)) (compute . core cmp) arr''
  where
    core c = unpairP . fmap c . riffle'
sort :: forall a . (Scalar a, Ord a) => SPull (Exp a) -> SPush Block (Exp a)
sort = divideAndConquer $ shexRev' compareSwap

sortObs :: (Scalar a, Ord a) => DPull (Exp a) -> DPush Grid (Exp a)
sortObs arr = asGridMap sort (splitUp 1024 arr)

divideAndConquer:: forall a . Data a => (forall  t . (Array (Push t), Compute t) => SPull a -> SPush t a) -> SPull  a -> SPush Block  a
divideAndConquer f arr = execBlock $ doIt (logLen - 1) arr
  where logLen = logBaseI 2 (len arr)
        doIt 0 a =
          do
            return  $ (f :: SPull a -> SPush Block a) a

        doIt n a | currLen > 1024 = blockBody
                 | currLen > 32  = warpBody
                 | otherwise     = threadBody
          where
            currLen = 2^(logLen - n)
            arrs = splitUp currLen a
            blockBody =
              do
                arr' <- compute
                        $ asBlockMap (f :: SPull a -> SPush Block a)
                        $ arrs
                doIt (n - 1) arr'
            warpBody =
              do
                arr' <- compute
                        $ asBlockMap (f :: SPull a -> SPush Warp a)
                        $ arrs
                doIt (n - 1) arr'
            threadBody =
              do
                arr' <- compute
                        $ asBlockMap (f :: SPull a -> SPush Thread a)
                        $ arrs
                doIt (n - 1) arr'

-- | runSortObs: run the given binary on the GPU
runSortObs ctx kern inputs size sorted =
  withCUDA' ctx $
  do
    useVector inputs $ \i ->
      allocaVector size $ \ o ->
      do
        fill o 0
        o <== (1,kern) <> i
        r <- peekCUDAVector o
        return r
