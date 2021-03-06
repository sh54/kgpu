package io.github.kgpu.kshader

import java.nio.ByteBuffer
import org.lwjgl.util.shaderc.Shaderc

actual object KShader {

    actual fun init() {}

    actual suspend fun compile(name: String, source: String, type: KShaderType): ByteArray {
        val compiler = Shaderc.shaderc_compiler_initialize()
        val options = Shaderc.shaderc_compile_options_initialize()

        val result: Long =
            Shaderc.shaderc_compile_into_spv(
                compiler, source, type.nativeType, name, "main", options)

        if (Shaderc.shaderc_result_get_compilation_status(result) !=
            Shaderc.shaderc_compilation_status_success) {
            var message = Shaderc.shaderc_result_get_error_message(result)
            if (message == null || message.isBlank())
                message = "error code " + Shaderc.shaderc_result_get_compilation_status(result)
            throw RuntimeException("Failed to compile shader: $message")
        }

        val output: ByteBuffer =
            Shaderc.shaderc_result_get_bytes(result)
                ?: throw RuntimeException("Failed to get results of shader compliation")
        val outputArray = ByteArray(output.remaining())
        output.get(outputArray)

        Shaderc.shaderc_result_release(result)
        Shaderc.shaderc_compile_options_release(options)
        Shaderc.shaderc_compiler_release(compiler)

        return outputArray
    }
}

actual enum class KShaderType(internal val nativeType: Int) {
    VERTEX(Shaderc.shaderc_vertex_shader),
    FRAGMENT(Shaderc.shaderc_fragment_shader),
    COMPUTE(Shaderc.shaderc_compute_shader)
}
