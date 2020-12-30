package io.github.kgpu.wgpuj.jni;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public enum WgpuPrimitiveTopology {
    /** Vertex data is a list of points. Each vertex is a new point. */
    POINT_LIST,
    /**
     * Vertex data is a list of lines. Each pair of vertices composes a new line.
     *
     * <p>Vertices `0 1 2 3` create two lines `0 1` and `2 3`
     */
    LINE_LIST,
    /**
     * Vertex data is a strip of lines. Each set of two adjacent vertices form a line.
     *
     * <p>Vertices `0 1 2 3` create three lines `0 1`, `1 2`, and `2 3`.
     */
    LINE_STRIP,
    /**
     * Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
     *
     * <p>Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
     */
    TRIANGLE_LIST,
    /**
     * Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
     *
     * <p>Vertices `0 1 2 3 4 5` creates four triangles `0 1 2`, `2 1 3`, `3 2 4`, and `4 3 5`
     */
    TRIANGLE_STRIP,
}