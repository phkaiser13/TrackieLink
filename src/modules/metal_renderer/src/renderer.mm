#include "metal_renderer/renderer.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <AppKit/AppKit.h> // For NSView

// This file provides the Objective-C++ implementation for the Metal renderer.
// It demonstrates key concepts:
// 1. A private Objective-C class to encapsulate all the Metal logic.
// 2. A C-API that wraps this class, providing a stable interface for C++.
// 3. Manual memory management (`CFBridgingRetain`/`Release`) for the C-API handles.

// --- Private MetalRenderer Class Definition ---

/**
 * @interface MetalRenderer
 * @brief An Objective-C class that manages all the state and objects
 *        required for rendering with Metal.
 */
@interface MetalRenderer : NSObject
- (instancetype)initWithView:(NSView*)view;
- (void)draw;
@end


// --- Private MetalRenderer Class Implementation ---

@implementation MetalRenderer {
    // Metal objects
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLBuffer> _vertexBuffer;

    // The layer provided by QuartzCore that Metal can draw into.
    CAMetalLayer* _metalLayer;
}

/**
 * @brief The designated initializer for the renderer.
 *
 * This method sets up the entire Metal rendering environment.
 */
- (instancetype)initWithView:(NSView*)view {
    self = [super init];
    if (self) {
        // 1. Get the default Metal device.
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            NSLog(@"[MetalRenderer] Metal is not supported on this device.");
            return nil;
        }

        // 2. Create the command queue.
        _commandQueue = [_device newCommandQueue];

        // 3. Set up the CAMetalLayer on the view.
        _metalLayer = [CAMetalLayer layer];
        _metalLayer.device = _device;
        _metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm; // A common pixel format.
        _metalLayer.framebufferOnly = YES; // Performance optimization.
        _metalLayer.frame = view.bounds;
        [view.layer addSublayer:_metalLayer];
        view.wantsLayer = YES; // An NSView must have wantsLayer set to YES to use a CAMetalLayer.

        // 4. Load the compiled shader library (default.metallib).
        NSError* error = nil;
        id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary:&error];
        if (!defaultLibrary) {
            NSLog(@"[MetalRenderer] Failed to load default shader library: %@", error.localizedDescription);
            return nil;
        }
        id<MTLFunction> vertexFunction = [defaultLibrary newFunctionWithName:@"vertex_main"];
        id<MTLFunction> fragmentFunction = [defaultLibrary newFunctionWithName:@"fragment_main"];

        // 5. Create the render pipeline state.
        MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.label = @"Simple Pipeline";
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = fragmentFunction;
        pipelineDescriptor.colorAttachments[0].pixelFormat = _metalLayer.pixelFormat;

        _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
        if (!_pipelineState) {
            NSLog(@"[MetalRenderer] Failed to create pipeline state: %@", error.localizedDescription);
            return nil;
        }

        // 6. Create vertex data for a simple triangle.
        static const float vertexData[] = {
             0.0f,  0.5f, 0.0f, 1.0f, // Top center
            -0.5f, -0.5f, 0.0f, 1.0f, // Bottom left
             0.5f, -0.5f, 0.0f, 1.0f  // Bottom right
        };
        _vertexBuffer = [_device newBufferWithBytes:vertexData length:sizeof(vertexData) options:MTLResourceStorageModeShared];
    }
    return self;
}

/**
 * @brief The main drawing method, called for each frame.
 */
- (void)draw {
    // Use an autorelease pool for any temporary Objective-C objects created during the frame.
    @autoreleasepool {
        // 1. Get the next available drawable from the layer.
        id<CAMetalDrawable> drawable = [_metalLayer nextDrawable];
        if (!drawable) {
            return; // Can happen if the view is not visible.
        }

        // 2. Create a render pass descriptor, which holds the rendering targets.
        MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.2, 1.0); // Dark blue background

        // 3. Create a command buffer to hold our rendering commands.
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        commandBuffer.label = @"MyFrameCommand";

        // 4. Create a render command encoder to encode the drawing commands.
        id<MTLRenderCommandEncoder> commandEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        commandEncoder.label = @"MyRenderEncoder";

        // 5. Encode the commands to draw the triangle.
        [commandEncoder setRenderPipelineState:_pipelineState];
        [commandEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
        [commandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

        [commandEncoder endEncoding];

        // 6. Schedule the command buffer to present the drawable.
        [commandBuffer presentDrawable:drawable];

        // 7. Commit the command buffer to the GPU for execution.
        [commandBuffer commit];
    }
}

@end


// --- C-API Implementation ---

// The C-style handle is a struct containing a pointer to the Objective-C object.
struct renderer_t {
    MetalRenderer* renderer;
};

renderer_t* renderer_create(void* view_handle) {
    if (!view_handle) return NULL;
    NSView* view = (__bridge NSView*)view_handle;

    renderer_t* handle = (renderer_t*)malloc(sizeof(renderer_t));
    if (!handle) return NULL;

    // The @autoreleasepool is important for managing memory in non-ARC C/C++ contexts.
    @autoreleasepool {
        handle->renderer = [[MetalRenderer alloc] initWithView:view];
        if (!handle->renderer) {
            free(handle);
            return NULL;
        }
        // We are transferring ownership from ARC to our C code.
        // We must now manage this object's lifetime manually.
        (void)CFBridgingRetain(handle->renderer);
    }
    return handle;
}

void renderer_destroy(renderer_t* renderer) {
    if (!renderer) return;
    @autoreleasepool {
        // This transfers ownership back to ARC, which will deallocate the object.
        MetalRenderer* obj = (__bridge_transfer MetalRenderer*)renderer->renderer;
        obj = nil; // Not strictly necessary, but good practice.
    }
    free(renderer);
}

void renderer_draw_frame(renderer_t* renderer) {
    if (!renderer || !renderer->renderer) return;
    @autoreleasepool {
        [renderer->renderer draw];
    }
}
