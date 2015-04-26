{-# LANGUAGE MultiParamTypeClasses ,NoMonomorphismRestriction ,TypeOperators ,TypeSynonymInstances ,FlexibleInstances ,ScopedTypeVariables #-}
module MandelObs where

import Obsidian
import Obsidian.CodeGen.CUDA
import Obsidian.Run.CUDA.Exec

import Data.Word
import Prelude hiding (zipWith,sum,replicate,take,drop,iterate)

-- Mandel
xmin, xmax, ymin, ymax :: EFloat
xmax = 0.83
xmin = -2.23
ymax =  1.15
ymin =  -1.15

-- For generating a 512x512 image
deltaP, deltaQ :: EFloat
deltaP = (xmax - xmin) / 1024.0
deltaQ = (ymax - ymin) / 1024.0

f :: EFloat -> EFloat -> (EFloat, EFloat, EWord32) -> (EFloat, EFloat, EWord32)
f b t (x,y,iter) =
  (xsq - ysq + (xmin + t * deltaP), 2*x*y + (ymax - b * deltaQ), iter+1)
  where
    xsq = x*x
    ysq = y*y

cond :: (EFloat, EFloat, EWord32) -> EBool
cond (x,y,iter) = ((xsq + ysq) <* 4) &&* iter <* 512
  where
    xsq = x*x
    ysq = y*y

iters :: EWord32 -> EWord32 -> Program Thread EW8
iters bid tid =
  do (_,_,c) <- seqUntil (f bid' tid') cond  (0,0,1)
     return (color c)
  where
    color c = (w32ToW8 (c `mod` 16)) * 16
    tid' = w32ToF tid
    bid' = w32ToF bid

genRect :: EWord32 -> Word32 -> (EWord32 -> EWord32 -> SPush Thread b) -> DPush Grid b
genRect bs ts p = asGrid
                $ mkPull bs
                $ \bid -> asBlock $ mkPull ts (p bid)

mandel width height = genRect width height body
  where
    body i j = execThread' (iters i j)

runMandelObs ctx kern threads size =
  withCUDA' ctx $
  do
    allocaVector (size) $ \o ->
      do
        o <== (fromIntegral threads,kern)
        r <- copyOut o
        return r
