import io.github.kgpu.*
import io.github.kgpu.kcgmath.*
import io.github.kgpu.kshader.*

object SkyboxShaders {
    const val VERTEX =
        """
        #version 450

        out gl_PerVertex {
            vec4 gl_Position;
        };
        layout(location = 0) out vec3 texCoords;
        
        layout(set = 0, binding = 0) uniform Data {
            mat4 proj;
            mat4 view;
        };
        
        void main() {
            vec4 pos = vec4(0.0);
            switch(gl_VertexIndex) {
                case 0: pos = vec4(-1.0, -1.0, 0.0, 1.0); break;
                case 1: pos = vec4(3.0, -1.0, 0.0, 1.0); break;
                case 2: pos = vec4(-1.0,  3.0, 0.0, 1.0); break;
            }
            mat3 invModelView = transpose(mat3(view));
            vec3 unProjected = (inverse(proj) * pos).xyz;
            texCoords = invModelView * unProjected;
        
            gl_Position = pos;
        }
    """

    const val FRAG =
        """
        #version 450
        
        layout(set = 0, binding = 1) uniform textureCube cubeTexture;
        layout(set = 0, binding = 2) uniform sampler cubeSampler;

        layout(location = 0) in vec3 texCoords;
        layout(location = 0) out vec4 outColor;
        
        void main() {
            outColor = texture(samplerCube(cubeTexture, cubeSampler), texCoords);
        }
    """
}

suspend fun runSkyboxExample(window: Window) {
    fun getProjectionMatrix(): Matrix4 {
        val windowSize = window.windowSize
        val aspectRatio = windowSize.width.toFloat() / windowSize.height

        return Matrix4().perspective(MathUtils.toRadians(45f), aspectRatio, 1f, 10f)
    }
    val viewMatrix = Matrix4().lookAt(Vec3(1.5f, -1f, 3f), Vec3.ZERO, Vec3.UNIT_Y)

    val adapter = Kgpu.requestAdapterAsync(window)
    val device = adapter.requestDeviceAsync()
    val queue = device.getDefaultQueue()

    val vertexShader =
        device.createShaderModule(
            KShader.compile("vertex", SkyboxShaders.VERTEX, KShaderType.VERTEX))
    val fragShader =
        device.createShaderModule(KShader.compile("frag", SkyboxShaders.FRAG, KShaderType.FRAGMENT))
    val uniformBuffer =
        BufferUtils.createBufferFromData(
            device,
            "Uniform Buffer",
            getProjectionMatrix().toBytes() + viewMatrix.toBytes(),
            BufferUsage.COPY_DST or BufferUsage.UNIFORM)

    val imageSize = 1024L
    val texture =
        device.createTexture(
            TextureDescriptor(
                Extent3D(imageSize, imageSize, 6),
                1,
                1,
                TextureDimension.D2,
                TEXTURE_FORMAT,
                TextureUsage.SAMPLED or TextureUsage.COPY_DST))
    arrayOf("front.png", "back.png", "up.png", "down.png", "right.png", "left.png")
        .map { name -> loadImage("/skybox/$name").second }
        .forEachIndexed { layer, bytes ->
            queue.writeTexture(
                TextureCopyView(texture, 0, Origin3D(0, 0, layer.toLong())),
                bytes,
                TextureDataLayout(4 * imageSize.toInt(), 0, 0L),
                Extent3D(imageSize, imageSize, 1))
            queue.submit()
        }
    val textureView =
        texture.createView(TextureViewDescriptor(TEXTURE_FORMAT, TextureViewDimension.CUBE))
    val sampler = device.createSampler(SamplerDescriptor())
    val bindGroupLayout =
        device.createBindGroupLayout(
            BindGroupLayoutDescriptor(
                BindGroupLayoutEntry(
                    0,
                    ShaderVisibility.VERTEX or ShaderVisibility.FRAGMENT,
                    BindingType.UNIFORM_BUFFER),
                BindGroupLayoutEntry(
                    1,
                    ShaderVisibility.FRAGMENT,
                    BindingType.SAMPLED_TEXTURE,
                    false,
                    TextureViewDimension.CUBE,
                    TextureComponentType.FLOAT),
                BindGroupLayoutEntry(2, ShaderVisibility.FRAGMENT, BindingType.SAMPLER, false),
            ))
    val bindGroup =
        device.createBindGroup(
            BindGroupDescriptor(
                bindGroupLayout,
                BindGroupEntry(0, uniformBuffer),
                BindGroupEntry(1, textureView),
                BindGroupEntry(2, sampler)))

    val pipelineLayout = device.createPipelineLayout(PipelineLayoutDescriptor(bindGroupLayout))
    val pipelineDesc =
        RenderPipelineDescriptor(
            pipelineLayout,
            ProgrammableStageDescriptor(vertexShader, "main"),
            ProgrammableStageDescriptor(fragShader, "main"),
            PrimitiveTopology.TRIANGLE_LIST,
            RasterizationStateDescriptor(),
            arrayOf(
                ColorStateDescriptor(
                    TextureFormat.BGRA8_UNORM, BlendDescriptor(), BlendDescriptor(), 0xF)),
            Kgpu.undefined,
            VertexStateDescriptor(null),
            1,
            0xFFFFFFFF,
            false)
    val pipeline = device.createRenderPipeline(pipelineDesc)
    val swapChainDescriptor = SwapChainDescriptor(device, TextureFormat.BGRA8_UNORM)

    var swapChain = window.configureSwapChain(swapChainDescriptor)
    window.onResize =
        { size: WindowSize ->
            swapChain = window.configureSwapChain(swapChainDescriptor)
        }

    Kgpu.runLoop(window) {
        val swapChainTexture = swapChain.getCurrentTextureView()
        val cmdEncoder = device.createCommandEncoder()

        val colorAttachment = RenderPassColorAttachmentDescriptor(swapChainTexture, Color.WHITE)
        val renderPassEncoder = cmdEncoder.beginRenderPass(RenderPassDescriptor(colorAttachment))
        renderPassEncoder.setPipeline(pipeline)
        renderPassEncoder.setBindGroup(0, bindGroup)
        renderPassEncoder.draw(3, 1)
        renderPassEncoder.endPass()

        val cmdBuffer = cmdEncoder.finish()
        viewMatrix.rotate(0f, 0.02f, 0f)
        queue.writeBuffer(uniformBuffer, viewMatrix.toBytes(), 64, 0, 64)
        queue.submit(cmdBuffer)
        swapChain.present()
    }
}
