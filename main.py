from jujutsu_engine import JujutsuEngine, InfiniteVoid , ReversalRed

def main():
    engine = JujutsuEngine(camera_index=0)
    
    void_expansion = InfiniteVoid("D:/interactive-media-engine/techniques/unlimited_void.mp4")
    reversal_red =  ReversalRed("D:/interactive-media-engine/techniques/reversal_red.mp4")
    
    engine.register_technique(void_expansion)
    engine.register_technique(reversal_red)
    
    engine.start()

if __name__ == "__main__":
    main()
