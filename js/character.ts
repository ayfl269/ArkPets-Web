import spine from '../libs/spine-webgl.js';
import webgl = spine.webgl;
import outlineFragmentShader from '../shaders/OutlineFragment.glsl';
import outlineVertexShader from '../shaders/OutlineVertex.glsl';
import { CharacterModel } from './types.js';

const MOVING_SPEED = 30; // pixels per second

const GRAVITY = 1000; // pixels per second squared
const DRAG = 0.98; // air resistance
const MAX_VELOCITY = 1000; // maximum velocity in pixels per second
const MIN_VELOCITY = 5; // threshold for stopping
const BOUNCE_DAMPING = 0.7; // energy loss on bounce

// Screen size breakpoints and corresponding canvas base sizes
const SCREEN_CONFIG = {
    MOBILE_BREAKPOINT: 480,
    MOBILE_BASE_SIZE: 160,
    DESKTOP_BASE_SIZE: 210,
    VIEWPORT_RATIO: 0.9
};

// Minimum and maximum scale factors
const minScale: number = 0.5;
const maxScale: number = 2.0;

const ANIMATION_NAMES = ["Relax", "Interact", "Move", "Sit" , "Sleep", "Special"];
const ANIMATION_MARKOV = [
    [0.5, 0.0, 0.2, 0.1, 0.1, 0.1],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.6, 0.0, 0.0, 0.2],
    [0.3, 0.0, 0.0, 0.5, 0.0, 0.2],
    [0.1, 0.0, 0.0, 0.0, 0.9, 0.0],
    [0.4, 0.0, 0.4, 0.1, 0.1, 0.0], 
]

// Vehicle can't sit & sleep
const ANIMATION_NAMES_VEHICLE = ["Relax", "Interact", "Move", "Special"];
const ANIMATION_MARKOV_VEHICLE = [
    [0.4, 0.0, 0.5, 0.1],
    [1.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.6, 0.1],
    [0.5, 0.0, 0.5, 0.0],
]

// Added: Create separate animation chain for characters without Special animation
const ANIMATION_NAMES_NO_SPECIAL = ["Relax", "Interact", "Move", "Sit", "Sleep"];
const ANIMATION_MARKOV_NO_SPECIAL = [
    [0.5, 0.0, 0.25, 0.15, 0.1],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.8, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.7, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.9],
]

// Vehicle without Special animation
const ANIMATION_NAMES_VEHICLE_NO_SPECIAL = ["Relax", "Interact", "Move"];
const ANIMATION_MARKOV_VEHICLE_NO_SPECIAL = [
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0],
    [0.3, 0.0, 0.7],
]

interface SpineCharacter {
    skeleton: spine.Skeleton;
    state: spine.AnimationState;
}

interface Action {
    animation: string;
    direction: Direction;
    timestamp: number;
}

type Direction = "left" | "right";

export class Character {
    private canvas!: HTMLCanvasElement;
    private gl!: WebGLRenderingContext;
    private shader!: webgl.Shader;
    private batcher!: webgl.PolygonBatcher;
    private mvp!: webgl.Matrix4;
    private assetManager!: webgl.AssetManager;
    private skeletonRenderer!: webgl.SkeletonRenderer;
    private lastFrameTime!: number;
    private framebuffer!: WebGLFramebuffer;
    private framebufferTexture!: WebGLTexture;
    private outlineShader!: WebGLProgram;
    private quadBuffer!: WebGLBuffer;

    private isMouseOver: boolean = false;
    
    // Dragging state
    private isDragging: boolean = false;
    private dragStartRelativeX: number = 0;
    private dragStartRelativeY: number = 0;
    private lastDragEvent: MouseEvent | null = null;
    
    // Physics state
    private velocity = { x: 0, y: 0 };
    
    private model: CharacterModel;
    private character!: SpineCharacter;
    
    private currentAction: Action = {
        animation: "Relax",
        direction: "right",
        timestamp: 0
    };
    
    private position: { x: number; y: number } = {
        x: -1, // will be set to a random value
        y: 1e9 // will be bounded to the bottom of the window
    };

    private animationFrameId: number | null = null;

    // Vehicle can't sit & sleep
    private isVehicle: boolean = false;
    
    // Added: Mark whether it has Special animation
    private hasSpecialAnimation: boolean = true;

    private allowInteract: boolean = true;

    // Supersampling is necessary for high-res display
    private pixelRatio!: number;
    
    // Event handler references for proper removal
    private handleMouseMoveRef: (event: MouseEvent) => void;
    private handleDragRef: (event: MouseEvent | TouchEvent) => void;
    private handleDragEndRef: (event: MouseEvent | TouchEvent) => void;
    private handleDragStartRef: (event: MouseEvent | TouchEvent) => void;
    private handleCanvasClickRef: (event: MouseEvent) => void;
    private onWindowResizeRef: () => void;
    private onBeforeUnloadRef: () => void;
    
    constructor(canvasId: string, onContextMenu: (e: MouseEvent | TouchEvent) => void, initialCharacter: CharacterModel, allowInteract: boolean = true) {
        this.allowInteract = allowInteract;
        this.model = initialCharacter;
        this.mvp = new webgl.Matrix4();
        this.pixelRatio = Math.max(2, window.devicePixelRatio || 1);
        
        // Initialize event handler references
        this.handleMouseMoveRef = this.handleMouseMove.bind(this);
        this.handleDragRef = this.handleDrag.bind(this);
        this.handleDragEndRef = this.handleDragEnd.bind(this);
        this.handleDragStartRef = this.handleDragStart.bind(this);
        this.handleCanvasClickRef = this.handleCanvasClick.bind(this);
        this.onWindowResizeRef = this.onWindowResize.bind(this);
        this.onBeforeUnloadRef = () => {
            // Force save on page unload
            this.lastSaveTime = 0; 
            this.saveToSessionStorage();
        };
        
        // Initialize canvas and WebGL
        this.initializeCanvas(canvasId);
        this.initializeWebGL();
        this.setupEventListeners(onContextMenu);
        
        // Load initial character
        this.loadFromSessionStorage();
        this.loadCharacterModel(this.model);

        window.addEventListener('beforeunload', this.onBeforeUnloadRef);
    }

    public destroy(): void {
        // Stop animation
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        // Remove event listeners
        document.removeEventListener('mousemove', this.handleMouseMoveRef);
        document.removeEventListener('mousemove', this.handleDragRef);
        document.removeEventListener('mouseup', this.handleDragEndRef);
        document.removeEventListener('touchmove', this.handleDragRef);
        document.removeEventListener('touchend', this.handleDragEndRef);
        window.removeEventListener('resize', this.onWindowResizeRef);
        window.removeEventListener('beforeunload', this.onBeforeUnloadRef);

        if (this.canvas) {
            this.canvas.removeEventListener('click', this.handleCanvasClickRef);
            this.canvas.removeEventListener('mousedown', this.handleDragStartRef);
            this.canvas.removeEventListener('touchstart', this.handleDragStartRef);
            
            // Remove canvas from DOM
            if (this.canvas.parentNode) {
                this.canvas.parentNode.removeChild(this.canvas);
            }
        }

        // Clean up Spine resources
        if (this.character && this.character.state) {
            this.character.state.clearTracks();
            this.character.state.clearListeners();
        }

        // Clean up WebGL resources
        if (this.gl) {
            this.releaseWebGLResources();
        }

        // Clear session storage if needed (optional, but keep it for completeness if requested)
        // sessionStorage.removeItem('arkpets-character-' + this.canvas.id);

        // Clear asset manager
        if (this.assetManager) {
            this.assetManager.dispose();
        }
    }

    private initializeCanvas(canvasId: string): void {
        this.canvas = document.createElement('canvas');
        this.canvas.classList.add("arkpets-canvas");
        this.canvas.id = canvasId;
        document.body.appendChild(this.canvas);
        this.canvas.style.pointerEvents = "none";
        
        // Add window resize listener
        window.addEventListener('resize', this.onWindowResizeRef);
    }

    private initializeWebGL(): void {
        this.gl = this.canvas.getContext("webgl", {
            alpha: true,
            premultipliedAlpha: false
        }) as WebGLRenderingContext;

        if (!this.gl) {
            throw new Error('WebGL is unavailable.');
        }

        // Set up WebGL context
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        
        this.initFramebuffer();
        
        // Create WebGL objects
        this.shader = webgl.Shader.newTwoColoredTextured(this.gl);
        this.batcher = new webgl.PolygonBatcher(this.gl);
        this.skeletonRenderer = new webgl.SkeletonRenderer(new webgl.ManagedWebGLRenderingContext(this.gl));
        this.assetManager = new webgl.AssetManager(this.gl);
    }

    private setupEventListeners(onContextMenu: (e: MouseEvent | TouchEvent) => void): void {
        // Track mouse position to decide mouse over
        document.addEventListener('mousemove', this.handleMouseMoveRef);

        if (this.allowInteract) {
            // React to click events
            this.canvas.addEventListener('click', this.handleCanvasClickRef);

            // Context menu
            this.canvas.addEventListener('contextmenu', onContextMenu);

            // Mouse events
            this.canvas.addEventListener('mousedown', this.handleDragStartRef);
            document.addEventListener('mousemove', this.handleDragRef);
            document.addEventListener('mouseup', this.handleDragEndRef);
            
            // Touch events
            this.canvas.addEventListener('touchstart', this.handleDragStartRef);
            document.addEventListener('touchmove', this.handleDragRef);
            document.addEventListener('touchend', this.handleDragEndRef);
        }
    }

    private initFramebuffer(): void {
        // Create and bind framebuffer
        this.framebuffer = this.gl.createFramebuffer()!;
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);

        // Create and bind texture
        this.framebufferTexture = this.gl.createTexture()!;
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.framebufferTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

        // Attach texture to framebuffer
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.framebufferTexture, 0);

        // Create quad buffer for second pass
        this.quadBuffer = this.gl.createBuffer()!;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  // Bottom left
             1, -1,  // Bottom right
            -1,  1,  // Top left
             1,  1   // Top right
        ]), this.gl.STATIC_DRAW);

        // Create and compile outline shader
        const vertexShader = this.gl.createShader(this.gl.VERTEX_SHADER)!;
        this.gl.shaderSource(vertexShader, outlineVertexShader);
        this.gl.compileShader(vertexShader);

        const fragmentShader = this.gl.createShader(this.gl.FRAGMENT_SHADER)!;
        this.gl.shaderSource(fragmentShader, outlineFragmentShader);
        this.gl.compileShader(fragmentShader);

        // Check compilation status
        if (!this.gl.getShaderParameter(vertexShader, this.gl.COMPILE_STATUS)) {
            console.error('Vertex shader compilation failed:', this.gl.getShaderInfoLog(vertexShader));
        }
        if (!this.gl.getShaderParameter(fragmentShader, this.gl.COMPILE_STATUS)) {
            console.error('Fragment shader compilation failed:', this.gl.getShaderInfoLog(fragmentShader));
        }
        
        this.outlineShader = this.gl.createProgram()!;
        this.gl.attachShader(this.outlineShader, vertexShader);
        this.gl.attachShader(this.outlineShader, fragmentShader);
        this.gl.linkProgram(this.outlineShader);
        
        if (!this.gl.getProgramParameter(this.outlineShader, this.gl.LINK_STATUS)) {
            console.error('Program linking failed:', this.gl.getProgramInfoLog(this.outlineShader));
        }
        
        // Clean up shaders as they're now part of the program
        this.gl.deleteShader(vertexShader);
        this.gl.deleteShader(fragmentShader);
    }

    public async loadCharacterModel(model: CharacterModel) {
        this.model = model;
        
        function encodeUriPath(path: string): string {
            return encodeURIComponent(path).replace(/%2F/g, '/');
        }
        console.log("Downloading character assets for", model.name);

        const resources = [model.skeleton, model.atlas, model.texture];
        const basePath = model.resourcePath ?? "";
        
        try {
            const objectUrls = await Promise.all(resources.map(async resource => {
                const response = await fetch(basePath + encodeUriPath(resource));
                if (!response.ok) throw new Error(`Failed to fetch ${resource}: ${response.statusText}`);
                const blob = await response.blob();
                return URL.createObjectURL(blob);
            }));

            resources.forEach((resource, i) => {
                this.assetManager.setRawDataURI(resource, objectUrls[i]);
            });

            this.assetManager.removeAll();
            
            // Load skeleton and atlas
            await new Promise<void>((resolve, reject) => {
                this.assetManager.loadBinary(model.skeleton, () => {
                    this.assetManager.loadTextureAtlas(model.atlas, () => {
                        console.log("Loaded character assets for", model.name);
                        // Cleanup object URLs and raw data URIs
                        resources.forEach((resource, i) => {
                            URL.revokeObjectURL(objectUrls[i]);
                            this.assetManager.setRawDataURI(resource, ""); 
                        });
                        resolve();
                    }, (error) => reject(new Error(`Failed to load atlas: ${error}`)));
                }, (error) => reject(new Error(`Failed to load skeleton: ${error}`)));
            });

            if (this.animationFrameId !== null) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
            requestAnimationFrame(this.load.bind(this));

        } catch (error) {
            console.error("Failed to load character assets:", error);
        }
    }

    public fadeOut(): Promise<void> {
        return new Promise((resolve) => {
            let opacity = 1;
            const fadeInterval = setInterval(() => {
                opacity -= 0.1;
                this.canvas.style.opacity = opacity.toString();
                
                if (opacity <= 0) {
                    clearInterval(fadeInterval);
                    resolve();
                }
            }, 30);
        });
    }

    private lastSaveTime: number = 0;
    private readonly SAVE_INTERVAL = 1000; // 1 second

    private saveToSessionStorage(): void {
        const now = Date.now();
        if (now - this.lastSaveTime < this.SAVE_INTERVAL) {
            return;
        }
        this.lastSaveTime = now;

        sessionStorage.setItem('arkpets-character-' + this.canvas.id, JSON.stringify({
            position: this.position,
            currentAction: this.currentAction,
            characterResource: this.model
        }));
    }

    private loadFromSessionStorage(): void {
        const saved = sessionStorage.getItem('arkpets-character-' + this.canvas.id);
        if (saved) {
            const state = JSON.parse(saved);
            this.position = state.position;
            this.currentAction = state.currentAction;
            this.model = state.characterResource;
        }
    }

    private load(): void {
        if (this.assetManager.isLoadingComplete()) {
            this.character = this.loadCharacter(this.model, 0.3);

            if (this.getAnimationNames().indexOf(this.currentAction.animation) === -1) {
                // If switching from character to vehicle, make sure it's not in `Sleep` or `Sit`
                this.currentAction.animation = "Relax";
                this.currentAction.timestamp = 0;
            }
            this.character.state.setAnimation(0, this.currentAction.animation, true);
            this.character.state.update(this.currentAction.timestamp);

            this.lastFrameTime = Date.now() / 1000;

            // Generate random x position if it's not set yet
            if (this.position.x === -1) {
                this.position.x = Math.random() * (window.innerWidth - this.canvas.offsetWidth);
            }

            requestAnimationFrame(this.render.bind(this));
        } else {
            console.debug("Loading assets of character", this.model.name, "progress", this.assetManager.getLoaded(), "/", this.assetManager.getToLoad());
            requestAnimationFrame(this.load.bind(this));
        }
    }

    private loadCharacter(resource: CharacterModel, scale: number = 1.0): SpineCharacter {    
        const atlas = this.assetManager.get(resource.atlas);
        const atlasLoader = new spine.AtlasAttachmentLoader(atlas);
        const skeletonBinary = new spine.SkeletonBinary(atlasLoader);

        skeletonBinary.scale = scale;
        const skeletonData = skeletonBinary.readSkeletonData(this.assetManager.get(resource.skeleton));
        const skeleton = new spine.Skeleton(skeletonData);

        if (!skeletonData.findAnimation("Sit") || !skeletonData.findAnimation("Sleep")) {
            this.isVehicle = true;
        }
        
        // Check if has Special animation
        this.hasSpecialAnimation = !!skeletonData.findAnimation("Special");

        const animationStateData = new spine.AnimationStateData(skeleton.data);

        // Animation transitions
        this.getAnimationNames().forEach(fromAnim => {
            this.getAnimationNames().forEach(toAnim => {
                if (fromAnim !== toAnim) {
                    animationStateData.setMix(fromAnim, toAnim, 0.3);
                }
            });
        });

        const animationState = new spine.AnimationState(animationStateData);
        animationState.setAnimation(0, "Relax", true);

        // Listen for animation completion
        const self = this;
        class AnimationStateAdapter extends spine.AnimationStateAdapter {
            complete(entry: spine.TrackEntry): void {
                const action = self.nextAction(self.currentAction);
                self.currentAction = action;
                console.debug("Play action", action)
                animationState.setAnimation(0, action.animation, true);
            }
        }
        animationState.addListener(new AnimationStateAdapter());

        // Calculate adaptive canvas size
        this.updateCanvasSize();

        // Scale up the skeleton position to match the higher resolution
        skeleton.x = this.canvas.width / 2;
        skeleton.y = 0;

        // Update framebuffer texture size
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.framebufferTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);

        return {
            skeleton,
            state: animationState,
        };
    }

    // Mouse position (client, no transform, no supersampling)
    private currentMousePos = { x: 0, y: 0 };

    private handleMouseMove(event: MouseEvent): void {
        this.currentMousePos.x = event.clientX;
        this.currentMousePos.y = event.clientY;
    }

    private frameCount: number = 0;
    private readonly HIT_TEST_INTERVAL = 5; // Check every 5 frames

    private lastMouseX: number = -1;
    private lastMouseY: number = -1;

    private render(): void {
        this.frameCount++;
        const now = Date.now() / 1000;
        const delta = now - this.lastFrameTime;
        this.lastFrameTime = now;
        this.currentAction.timestamp += delta;

        // Apply physics when not dragging
        if (!this.isDragging) {
            // Apply gravity
            this.velocity.y += GRAVITY * delta;
            
            // Apply drag
            this.velocity.x *= DRAG;
            this.velocity.y *= DRAG;
            if (Math.abs(this.velocity.x) < MIN_VELOCITY) {
                this.velocity.x = 0;
            }
            if (Math.abs(this.velocity.y) < MIN_VELOCITY) {
                this.velocity.y = 0;
            }
            
            // Clamp velocities
            this.velocity.x = Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.velocity.x));
            this.velocity.y = Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.velocity.y));
            
            // Update position
            this.position.x += this.velocity.x * delta;
            this.position.y += this.velocity.y * delta;
            
            // Window bounds collision
            const maxX = window.innerWidth - this.canvas.offsetWidth;
            const maxY = window.innerHeight - this.canvas.offsetHeight;
            
            // Bounce off walls
            if (this.position.x < 0) {
                this.position.x = 0;
                this.velocity.x = -this.velocity.x * BOUNCE_DAMPING;
            } else if (this.position.x > maxX) {
                this.position.x = maxX;
                this.velocity.x = -this.velocity.x * BOUNCE_DAMPING;
            }
            
            // Bounce off floor/ceiling
            if (this.position.y < 0) {
                this.position.y = 0;
                this.velocity.y = 0;
            } else if (this.position.y > maxY) {
                this.position.y = maxY;
                this.velocity.y = 0;
            }
        }

        // Move the canvas when "Move" animation is playing
        if (this.currentAction.animation === "Move") {
            const movement = MOVING_SPEED * delta;
            if (this.currentAction.direction === "left") {
                this.position.x = Math.max(0, this.position.x - movement);
                // Turn around when reaching left edge
                if (this.position.x <= 0) {
                    this.position.x = 0;
                    this.currentAction.direction = "right";
                }
            } else {
                this.position.x = this.position.x + movement;
                // Turn around when reaching right edge
                if (this.position.x >= window.innerWidth - this.canvas.offsetWidth) {
                    this.position.x = window.innerWidth - this.canvas.offsetWidth;
                    this.currentAction.direction = "left";
                }
            }
        }

        // Update canvas position to `position`
        this.canvas.style.transform = `translate(${this.position.x}px, ${this.position.y}px)`;

        // 1st pass - render Spine character to framebuffer
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0, 0, 0, 0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        
        this.character.skeleton.scaleX = (this.currentAction.direction === "left" ? -1 : 1) * this.pixelRatio;
        this.character.skeleton.scaleY = this.pixelRatio;

        this.character.state.update(delta);
        this.character.state.apply(this.character.skeleton);
        this.character.skeleton.updateWorldTransform();

        this.shader.bind();
        this.shader.setUniformi(webgl.Shader.SAMPLER, 0);
        this.shader.setUniform4x4f(webgl.Shader.MVP_MATRIX, this.mvp.values);

        this.batcher.begin(this.shader);
        this.skeletonRenderer.premultipliedAlpha = false;
        this.skeletonRenderer.draw(this.batcher, this.character.skeleton);
        this.batcher.end();

        this.shader.unbind();

        // Read pixels before 2nd pass to determine if mouse is over character
        // Throttle hit testing and only check if mouse moved or character moved
        const mouseMoved = this.currentMousePos.x !== this.lastMouseX || this.currentMousePos.y !== this.lastMouseY;
        if (this.frameCount % this.HIT_TEST_INTERVAL === 0 || mouseMoved) {
            this.lastMouseX = this.currentMousePos.x;
            this.lastMouseY = this.currentMousePos.y;

            const canvasRect = this.canvas.getBoundingClientRect();
            let pixelX = (this.currentMousePos.x - canvasRect.x) * this.pixelRatio;
            let pixelY = this.canvas.height - (this.currentMousePos.y - canvasRect.y) * this.pixelRatio;
            
            if (pixelX >= 0 && pixelX < this.canvas.width && pixelY >= 0 && pixelY < this.canvas.height) {
                // Mouse inside canvas. Check whether it's over the character
                let pixelColor = new Uint8Array(4);
                this.gl.readPixels(
                    pixelX, 
                    pixelY, 
                    1, 1, 
                    this.gl.RGBA, 
                    this.gl.UNSIGNED_BYTE, 
                    pixelColor
                );
                this.isMouseOver = pixelColor[3] !== 0;
                if (this.allowInteract) {
                    this.canvas.style.pointerEvents = this.isMouseOver ? 'auto' : 'none';
                } else {
                    this.canvas.style.pointerEvents = 'none';
                }
            } else {
                this.isMouseOver = false;
                this.canvas.style.pointerEvents = 'none';
            }
        }

        // 2nd pass - render to screen with outline effect
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0, 0, 0, 0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.useProgram(this.outlineShader);

        // Set uniforms
        const uTexture = this.gl.getUniformLocation(this.outlineShader, "u_texture");
        const uOutlineColor = this.gl.getUniformLocation(this.outlineShader, "u_outlineColor");
        const uOutlineWidth = this.gl.getUniformLocation(this.outlineShader, "u_outlineWidth");
        const uTextureSize = this.gl.getUniformLocation(this.outlineShader, "u_textureSize");
        const uAlpha = this.gl.getUniformLocation(this.outlineShader, "u_alpha");

        this.gl.uniform1i(uTexture, 0); // Use texture unit 0 for spine character
        this.gl.uniform4f(uOutlineColor, 1.0, 1.0, 0.0, 1.0); // yellow
        this.gl.uniform1f(uOutlineWidth, this.allowInteract && this.isMouseOver ? 2.0 * this.pixelRatio : 0.0); // Show outline only in interactive mode
        this.gl.uniform2i(uTextureSize, this.canvas.width, this.canvas.height);
        this.gl.uniform1f(uAlpha, (!this.allowInteract && this.isMouseOver) ? 0.3 : 1.0); // Reduce opacity when non-interactive and mouse over

        // Bind framebuffer texture
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.framebufferTexture);

        // Draw quad to canvas
        const aPosition = this.gl.getAttribLocation(this.outlineShader, "a_position");
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.enableVertexAttribArray(aPosition);
        this.gl.vertexAttribPointer(aPosition, 2, this.gl.FLOAT, false, 0, 0);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        this.saveToSessionStorage();

        // Store the animation frame ID
        this.animationFrameId = requestAnimationFrame(this.render.bind(this));
    }

    private randomPick(probabilities: number[]): number {
        let random = Math.random();
        let cumulativeProb = 0;
        for (let i = 0; i < probabilities.length; i++) {
            cumulativeProb += probabilities[i];
            if (random <= cumulativeProb) {
                return i;
            }
        }
        throw new Error("Invalid probabilities: " + probabilities);
    }

    private turnDirection(current: Direction): Direction {
        return current === "left" ? "right" : "left";
    }

    private nextAction(current: Action): Action {
        const animeIndex = this.getAnimationNames().indexOf(current.animation);
        const nextIndexProb = this.getAnimationMarkov()[animeIndex];
        const nextAnimIndex = this.randomPick(nextIndexProb);
        const nextAnim = this.getAnimationNames()[nextAnimIndex];

        let nextDirection = current.direction;
        if (current.animation === "Relax" && nextAnim === "Move") {
            nextDirection = Math.random() < 0.4 ? this.turnDirection(current.direction) : current.direction;
        }
        return {
            animation: nextAnim,
            direction: nextDirection,
            timestamp: 0
        };
    }

    private handleCanvasClick(): void {
        if (this.character && this.character.state) {
            this.currentAction = {
                animation: "Interact",
                direction: this.currentAction.direction,
                timestamp: 0,
            };
            this.character.state.setAnimation(0, "Interact", false);
            console.debug("Play action", this.currentAction);
        }
    }

    private handleDragStart(e: MouseEvent | TouchEvent): void {
        if ((e as MouseEvent).button === undefined || (e as MouseEvent).button === 0) {
            this.isDragging = true;
            
            const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
            const clientY = 'touches' in e ? e.touches[0].clientY : (e as MouseEvent).clientY;
            this.dragStartRelativeX = clientX - this.position.x;
            this.dragStartRelativeY = clientY - this.position.y;
            
            // Pause any current animation
            if (this.character && this.character.state) {
                this.character.state.setAnimation(0, "Relax", true);
                this.currentAction = {
                    animation: "Relax",
                    direction: this.currentAction.direction,
                    timestamp: 0
                };
            }
        }
    }

    private handleDrag(e: MouseEvent | TouchEvent): void {
        if (this.isDragging) {
            const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
            const clientY = 'touches' in e ? e.touches[0].clientY : (e as MouseEvent).clientY;
            
            const oldX = this.position.x;
            const oldY = this.position.y;
            const newX = clientX - this.dragStartRelativeX;
            const newY = clientY - this.dragStartRelativeY;
            
            // Calculate velocity based on time between events
            if (this.lastDragEvent) {
                const dt = (e.timeStamp - this.lastDragEvent.timeStamp) / 1000;
                if (dt > 0) {
                    this.velocity.x = (newX - oldX) / dt;
                    this.velocity.y = (newY - oldY) / dt;
                }
            }
            
            // Update position
            this.position.x = newX;
            this.position.y = newY;
            this.canvas.style.transform = `translate(${this.position.x}px, ${this.position.y}px)`;
            
            this.lastDragEvent = e as MouseEvent;
            
            // Prevent scrolling on mobile
            if ('touches' in e) {
                e.preventDefault();
            }
        }
    }

    private handleDragEnd(): void {
        this.isDragging = false;
        this.lastDragEvent = null;
    }

    public getAnimationNames(): string[] {
        // If it's a vehicle type
        if (this.isVehicle) {
            // If it has Special animation
            if (this.hasSpecialAnimation) {
                return ANIMATION_NAMES_VEHICLE;
            } else {
                // If it doesn't have Special animation
                return ANIMATION_NAMES_VEHICLE_NO_SPECIAL;
            }
        } else {
            // If it's not a vehicle type
            // If it has Special animation
            if (this.hasSpecialAnimation) {
                return ANIMATION_NAMES;
            } else {
                // If it doesn't have Special animation
                return ANIMATION_NAMES_NO_SPECIAL;
            }
        }
    }

    private getAnimationMarkov(): number[][] {
        // Select corresponding transition matrix based on whether it's a vehicle and whether it has Special animation
        if (this.isVehicle) {
            // If it has Special animation
            if (this.hasSpecialAnimation) {
                return ANIMATION_MARKOV_VEHICLE;
            } else {
                // If it doesn't have Special animation
                return ANIMATION_MARKOV_VEHICLE_NO_SPECIAL;
            }
        } else {
            // If it's not a vehicle type
            // If it has Special animation
            if (this.hasSpecialAnimation) {
                return ANIMATION_MARKOV;
            } else {
                // If it doesn't have Special animation
                return ANIMATION_MARKOV_NO_SPECIAL;
            }
        }
    }

    public playAnimation(animationName: string): void {
        this.currentAction = {
            animation: animationName,
            direction: this.currentAction.direction,
            timestamp: 0,
        };
        this.character.state.setAnimation(0, animationName, true);
        console.debug("Play action", this.currentAction);
    }

    public getCanvasId(): string {
        return this.canvas.id;
    }

    public getModel(): CharacterModel {
        return this.model;
    }

    public isAllowInteract(): boolean {
        return this.allowInteract;
    }

    public setAllowInteract(allowInteract: boolean): void {
        this.allowInteract = allowInteract;
    }

    /**
     * Update canvas size based on window size and device pixel ratio
     * Implement high-resolution screen adaptation and maintain aspect ratio
     */
    private updateCanvasSize(): void {
        const { innerWidth, innerHeight } = window;
        
        // Dynamically calculate base size based on current window dimensions
        const isMobile = innerWidth <= SCREEN_CONFIG.MOBILE_BREAKPOINT;
        const maxBaseSize = isMobile ? SCREEN_CONFIG.MOBILE_BASE_SIZE : SCREEN_CONFIG.DESKTOP_BASE_SIZE;
        
        const baseWidth = Math.min(innerWidth * SCREEN_CONFIG.VIEWPORT_RATIO, maxBaseSize);
        const baseHeight = Math.min(innerHeight * SCREEN_CONFIG.VIEWPORT_RATIO, maxBaseSize);
        
        // In this project, baseWidth and baseHeight are usually the same (square)
        const aspectRatio = baseWidth / baseHeight;
        
        // Calculate best size to fit window
        let bestWidth = innerWidth;
        let bestHeight = innerHeight;
        
        if (innerWidth / innerHeight > aspectRatio) {
            // Window is wider, use height as reference
            bestHeight = innerHeight * 0.8;
            bestWidth = bestHeight * aspectRatio;
        } else {
            // Window is taller, use width as reference
            bestWidth = innerWidth * 0.8;
            bestHeight = bestWidth / aspectRatio;
        }
        
        // Apply scale limits (ensure between minScale and maxScale)
        const scale = Math.max(minScale, Math.min(maxScale, Math.min(bestWidth / baseWidth, bestHeight / baseHeight)));
        const finalWidth = baseWidth * scale;
        const finalHeight = baseHeight * scale;

        // Set canvas display size
        this.canvas.style.width = `${finalWidth}px`;
        this.canvas.style.height = `${finalHeight}px`;
        
        // Set canvas internal resolution (consider device pixel ratio)
        this.canvas.width = finalWidth * this.pixelRatio;
        this.canvas.height = finalHeight * this.pixelRatio;
        
        // Update projection matrix and WebGL viewport
        this.mvp.ortho2d(0, 0, this.canvas.width, this.canvas.height);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Window resize event handler
     */
    private onWindowResize(): void {
        // Update pixel ratio for zoom/high-res support
        this.pixelRatio = Math.max(2, window.devicePixelRatio || 1);

        // Dynamically adjust canvas size
        this.updateCanvasSize();
        
        // Only update framebuffer texture size, not recreate the entire framebuffer
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.framebufferTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        
        // Update character position to fit new dimensions
        this.position.x = Math.min(this.position.x, window.innerWidth - this.canvas.offsetWidth);
        this.position.y = Math.min(this.position.y, window.innerHeight - this.canvas.offsetHeight);
        
        // Update skeleton position and projection matrix
        if (this.character && this.character.skeleton) {
            this.character.skeleton.x = this.canvas.width / 2;
            this.character.skeleton.y = 0;
            // Update projection matrix
            this.mvp.ortho2d(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    /**
     * Release WebGL resources
     */
    private releaseWebGLResources(): void {
        if (this.framebuffer) {
            this.gl.deleteFramebuffer(this.framebuffer);
            this.framebuffer = null as any;
        }
        if (this.framebufferTexture) {
            this.gl.deleteTexture(this.framebufferTexture);
            this.framebufferTexture = null as any;
        }
        if (this.quadBuffer) {
            this.gl.deleteBuffer(this.quadBuffer);
            this.quadBuffer = null as any;
        }
        if (this.outlineShader) {
            this.gl.deleteProgram(this.outlineShader);
            this.outlineShader = null as any;
        }
    }
}