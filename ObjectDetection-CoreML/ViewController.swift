//
//  ViewController.swift
//  SSDMobileNet-CoreML
//
//  Created by GwakDoyoung on 01/02/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit
import ARKit
import RealityKit
import SceneKit.ModelIO
import Vision
import CoreMedia

class ViewController: UIViewController , ARSCNViewDelegate{

    // MARK: - UI Properties
    @IBOutlet weak var boxesView: DrawingBoundingBoxView!
    @IBOutlet weak var sceneView: ARSCNView!
    
    let nodeFlag = false
    let objectDectectionModel = YOLOv3()
    let depthDetectionModel = DeepLabV3()
    let maxPlanesCount = 5
    var currentPlaneCount = 0
    
    
    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var isInferencing = false
    
    // MARK: - AV Property
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    let planesRootNode = SCNNode()
    
    // MARK: - TableView Data
    var predictions: [VNRecognizedObjectObservation] = []
    
    // MARK - Performance Measurement Property
    //private let 👨‍🔧 = 📏()
    
    let configuration = ARWorldTrackingConfiguration()
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        let vaseScene = SCNScene(named:"IronMan/IronMan.scn")
        guard let node =  vaseScene?.rootNode else { return }
//        guard let modelScene = SCNScene(mdlAsset: mdlAsset),
//            let nodeModel =  modelScene.rootNode.childNode(
//               withName: "vase", recursively: true)
//        else{
//            print("fails")
//            return}

        
        sceneView.delegate = self
//
//         Show statistics such as fps and timing information
        sceneView.showsStatistics = true
//
//         Create a new scene
       let scene = SCNScene()
//
//         Set the scene to the view
        sceneView.scene = scene
        sceneView.scene.rootNode.addChildNode(node)
        sceneView.scene.rootNode.addChildNode(planesRootNode)

//
//         Enable Default Lighting - makes the 3D text a bit poppier.
        sceneView.autoenablesDefaultLighting = true
        // setup the model
        setUpModel()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    
        self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints,ARSCNDebugOptions.showWorldOrigin]
        self.sceneView.session.run(configuration)

        self.loopCoreMLUpdate()
        //self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        //self.videoCapture.stop()
    }
    
    func loopCoreMLUpdate() {
        // Continuously run CoreML whenever it's ready. (Preventing 'hiccups' in Frame Rate)
        
        DispatchQueue.main.async {
            // 1. Run Update.
            self.updateCoreML()
            
            // 2. Loop this function.
            self.loopCoreMLUpdate()
        }
        
    }
    
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: objectDectectionModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("fail to create vision model")
        }
    }
}


extension ViewController {
    func predictUsingVision(imageFromArkitScene: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        self.semaphore.wait()
        let handler = VNImageRequestHandler(cvPixelBuffer: imageFromArkitScene, orientation: .right)
        try? handler.perform([request])

    }
    
    
    // MARK: - Post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        //self.👨‍🔧.🏷(with: "endInference")
        if let predictions = request.results as? [VNRecognizedObjectObservation] {
            self.predictions = predictions
//            if self.sceneView.scene.rootNode.childNodes.count > 0
//            {
//                sceneView.scene.rootNode.enumerateChildNodes { (node, stop) in
//                    node.removeFromParentNode()
//                }
//            }

//            DispatchQueue.main.async {
                self.boxesView.subviews.forEach({ $0.removeFromSuperview() })
                for prediction in self.predictions{
                    let rect: CGRect = self.boxesView.createLabelAndBox(prediction: prediction)
                    self.addPlane(rect: rect)
                }
                self.isInferencing = false
//            }
        } else {
            
            self.isInferencing = false
        }
        self.semaphore.signal()
    }
    
    func addPlane(rect : CGRect){

        let screenCentre : CGPoint = CGPoint(x: rect.midX, y: rect.midY)
        let planeorigin : CGPoint = rect.origin
        let rightbottom : CGPoint = CGPoint(x: rect.maxX, y: rect.maxY)
        let centerTestResults : [ARHitTestResult] = sceneView.hitTest(screenCentre, types: [.featurePoint]) // Alternatively, we could use '.existingPlaneUsingExtent' for more grounded hit-test-points.
        let originTestResults : [ARHitTestResult] = sceneView.hitTest(planeorigin, types: [.featurePoint])
        let rightbotTestResults : [ARHitTestResult] = sceneView.hitTest(rightbottom, types: [.featurePoint])
        if let closestResult = centerTestResults.first {
            if let originResult = originTestResults.first{
                if let rightbotResult = rightbotTestResults.first{
                    
                    
                    let originTransform : matrix_float4x4 = originResult.worldTransform
                    let rightbotTransform : matrix_float4x4 = rightbotResult.worldTransform
                    let width : CGFloat = CGFloat(abs(rightbotTransform.columns.3.x - originTransform.columns.3.x))
                    let height : CGFloat = CGFloat(abs(originTransform.columns.3.y - rightbotTransform.columns.3.y))
                    print("the height and width:")
                    print(width)
                    print(height)
                    
                    let transform : matrix_float4x4 = closestResult.worldTransform
                    let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, (transform.columns.3.z + originTransform.columns.3.z + rightbotTransform.columns.3.z)/3.0)
                    // Create 3D Text
                    let node : SCNNode = createPlane(rect: rect, coordinate: worldCoord, planeWidth: width, planeHeight: height)
                    //node.geometry?.firstMaterial = maskMaterial()
                    
                    
                    if planesRootNode.childNodes.count > maxPlanesCount
                    {
                        planesRootNode.replaceChildNode(planesRootNode.childNodes[currentPlaneCount], with: node)
                        currentPlaneCount += 1
                        currentPlaneCount %= maxPlanesCount
                    }
                    else
                    {
                        planesRootNode.addChildNode(node)
                    }
                }
            }
            else{
                
                let transform : matrix_float4x4 = closestResult.worldTransform
                let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                // Create 3D Text
                let node : SCNNode = createPlane(rect: rect, coordinate: worldCoord, planeWidth: 0.2, planeHeight: 0.2)
                //node.geometry?.firstMaterial = maskMaterial()


                if planesRootNode.childNodes.count > maxPlanesCount
                {
                     planesRootNode.replaceChildNode(planesRootNode.childNodes[currentPlaneCount], with: node)
                    currentPlaneCount += 1
                    currentPlaneCount %= maxPlanesCount
                }
                else
                {
                    planesRootNode.addChildNode(node)
                }

            }
        }
    }
    
    func createPlane(rect : CGRect, coordinate: SCNVector3, planeWidth : CGFloat, planeHeight : CGFloat) -> SCNNode
    {
        let plane = SCNPlane(width: planeWidth, height: planeHeight)
        plane.cornerRadius = 0.005
        let planeNode = SCNNode(geometry: plane)
        planeNode.geometry?.firstMaterial?.isDoubleSided = true
        planeNode.geometry?.firstMaterial?.colorBufferWriteMask = .alpha
        planeNode.geometry?.firstMaterial?.writesToDepthBuffer = true
        planeNode.geometry?.firstMaterial?.readsFromDepthBuffer = true
        planeNode.renderingOrder = -100
//        planeNode.geometry?.firstMaterial = maskMaterial()
//        
//        planeNode.physicsBody = SCNPhysicsBody(type: .static,
//                                              shape: nil)
        planeNode.position = coordinate
        return planeNode
    }
    
    func maskMaterial() -> SCNMaterial {
        let maskMaterial = SCNMaterial()
        maskMaterial.diffuse.contents = UIColor.white
        
        // another way to do this is to set a very very low transparency value (but that
        // would not receive shadows..)
        // mask out everything we would have drawn..
        //maskMaterial.colorBufferWriteMask = SCNColorMask(rawValue: 0)
        
        // occlude (render) from both sides please
        maskMaterial.isDoubleSided = true
        return maskMaterial
    }
    //MARK: - updatePredictionByARscene
    func updateCoreML() {
        ///////////////////////////
        // Get Camera Image as RGB
        if let frame = sceneView.session.currentFrame{
            let imageBuffer = frame.capturedImage
            self.predictUsingVision(imageFromArkitScene: imageBuffer)

            
        }



    }
}



