//
//  my_SCNNode.swift
//  ObjectDetection-CoreML
//
//  Created by bowen liu on 3/30/20.
//  Copyright © 2020 tucan9389. All rights reserved.
//
import SceneKit.ModelIO

class MYNode{
    
    public var myNode: SCNNode
    
    public var myUID: Int32
    
    public var myDepth: Double
    
    init(){
        
        self.myNode  = SCNNode()
        self.myUID = -1
        self.myDepth = -1.0
        
    }
    
    init(node:SCNNode){
        
        self.myNode  = node
        self.myUID = -1
        self.myDepth = -1.0
        
    }
    
    init(node:SCNNode, id: Int32, depth: Double){
        
        self.myNode = node
        self.myUID = id
        self.myDepth = depth
        
    }
}
