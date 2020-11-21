package io.github.kgpu.wgpuj.jni;


/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public enum WgpuAddressMode {
     /**
       * Clamp the value to the edge of the texture
       *
       * -0.25 -> 0.0
       * 1.25  -> 1.0
       */
    CLAMP_TO_EDGE,
     /**
       * Repeat the texture in a tiling fashion
       *
       * -0.25 -> 0.75
       * 1.25 -> 0.25
       */
    REPEAT,
     /**
       * Repeat the texture, mirroring it every repeat
       *
       * -0.25 -> 0.25
       * 1.25 -> 0.75
       */
    MIRROR_REPEAT,
     /**
       * Clamp the value to the border of the texture
       * Requires feature [`Features::ADDRESS_MODE_CLAMP_TO_BORDER`]
       *
       * -0.25 -> border
       * 1.25 -> border
       */
    CLAMP_TO_BORDER,
}