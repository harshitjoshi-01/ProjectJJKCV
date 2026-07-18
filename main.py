from jujutsu_engine import JujutsuEngine, InfiniteVoid

def main():
    engine = JujutsuEngine(camera_index=0)
    
    void_expansion = InfiniteVoid("D:/interactive-media-engine/techniques/unlimited_void.mp4")
    
    engine.register_technique(void_expansion)
    
    engine.start()

if __name__ == "__main__":
    main()
