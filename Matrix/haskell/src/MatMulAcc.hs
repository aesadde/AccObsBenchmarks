{-# LANGUAGE TypeOperators #-}
module MatMulAcc(accMatMul) where

import Prelude as P
import Data.Array.Accelerate as A hiding ((++))
import Data.Array.Accelerate.CUDA as A

accMatMul a b = A.run $ sum' (prod aCube bCube)
  where
    sum' = A.fold (+) 0
    prod = A.zipWith (*)
    t    = A.transpose b
    getRow = A.indexHead . A.indexTail
    getCol = A.indexHead
    rowsA = getRow (A.shape a)
    colsB = getCol (A.shape b)
    sliceA = lift (Z :. All :. colsB :. All)
    sliceB = lift (Z :. rowsA :. All :. All)
    aCube = A.replicate sliceA a
    bCube = A.replicate sliceB t
