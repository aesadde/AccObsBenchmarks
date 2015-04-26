module Common (View, Bitmap, Render, RGBA32, prettyRGBA,) where

import Data.Array.Accelerate                    as A
import Data.Array.Accelerate.IO                 as A

-- Types -----------------------------------------------------------------------
-- Current view into the complex plane
type View a             = (a, a, a, a)
-- Image data
type Bitmap             = Array DIM2 RGBA32
-- Action to render a frame
type Render a           = Scalar (View a) -> Bitmap

-- Rendering -------------------------------------------------------------------
prettyRGBA :: Exp Int32 -> Exp Int32 -> Exp RGBA32
prettyRGBA cmax c = c ==* cmax ? ( 0xFF000000, escapeToColour (cmax - c) )

-- Directly convert the iteration count on escape to a colour. The base set
-- (x,y,z) yields a dark background with light highlights.
--
escapeToColour :: Exp Int32 -> Exp RGBA32
escapeToColour m = constant 0xFFFFFFFF - (packRGBA32 $ lift (x,y,z,w))
  where
    x   = constant 0
    w   = A.fromIntegral (3 * m)
    z   = A.fromIntegral (5 * m)
    y   = A.fromIntegral (7 * m)
