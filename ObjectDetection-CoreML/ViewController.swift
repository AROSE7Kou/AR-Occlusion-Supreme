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

let max_segmentation_count = 21

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {

    // MARK: - UI Properties
    @IBOutlet weak var sceneView: ARSCNView!
    var maskNodes = Array(repeating: MYNode.init(), count: max_segmentation_count)
    var maskMaterial : SCNMaterial!
    var id: Int32 = 40
    var registeredModels: [Int32 : MYNode] = [:]
    
    // MARK: - Vision Requests
    var currentBuffer: CVPixelBuffer?
    var visionRequestSeg = [VNRequest]()
    var visionRequestDepth = [VNRequest]()
    let visionQueue = DispatchQueue(label: "com.vision.ARML.visionqueue")
    
    // MARK: - ML Models
    var depthThreshold: Float = 0.0
    var depthMap: MLMultiArray? = nil
    var segmentationMap: MLMultiArray? = nil
    
    // MARK: - AV Property
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    let planesRootNode = SCNNode()
    var hitNodeResults: [SCNHitTestResult?] = []
    var draggingNode: SCNNode!
    var dragId : Int32!
    var cameraInverse: simd_float4x4?
    // MARK: - Button Controller
    var typeButton = -1
    var hStack: UIStackView! = UIStackView()
    var furniture1:UIButton! = UIButton()
    var furniture2:UIButton! = UIButton()
    var trash: UIImageView! = UIImageView()
    var detail1: UIView! = UIView()
    var detail2: UIView! = UIView()
    var info1: UITextView! = UITextView()
    var info2: UITextView! = UITextView()
    var infoindex = 0
    var infoarray = infospill()
    let models:[String] = ["3D Objects/table0.scn", "3D Objects/table1.scn", "3D Objects/desk0.scn", "3D Objects/desk1.scn", "3D Objects/cup1/cup1.scn", "3D Objects/cup2/cup2.scn", "3D Objects/chair1/chair1.scn", "3D Objects/chair2/chair2.scn", "3D Objects/sofa0.scn", "3D Objects/sofa1.scn"]
    
    // MARK - Performance Measurement Property
    //private let 👨‍🔧 = 📏()
    
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set Delegate
        sceneView.delegate = self
        sceneView.session.delegate = self
        
        sceneView.showsStatistics = true
        
//      Enable Default Lighting - makes the 3D text a bit poppier.
        sceneView.autoenablesDefaultLighting = true
        addTapGestureToSceneView()
        addPanGestureToSceneView()
        setUpModel()
//      create occlusion mask planes
        bilbordsCreate()
        
        // setup mask button
        let buttonMask = UIButton(frame: CGRect(x: 10, y: 50, width: 100, height: 30))
        buttonMask.backgroundColor = .gray
        buttonMask.titleLabel?.font =  UIFont.boldSystemFont(ofSize: 10)
        buttonMask.setTitle("Show mask", for: .normal)
        buttonMask.addTarget(self, action: #selector(buttonAction), for: .touchUpInside)
        self.view.addSubview(buttonMask)

        // setup furniture buttons
        furniture1 = {
            let button = UIButton()
            button.frame = CGRect(x:103, y:638, width:207, height:124)
            return button
        }()
        furniture2 = {
            let button = UIButton()
            button.frame = CGRect(x:309, y:638, width:207, height:124)
            return button
        }()
        
        furniture1.addTarget(self, action: #selector(fuck(sender:)), for: .touchDown)
        furniture1.addTarget(self, action: #selector(fuckme(sender:)), for: .touchUpInside)
        furniture1.addTarget(self, action: #selector(realfuckme(sender:)), for: .touchDownRepeat)
        furniture1.tag = 1
        
        furniture2.addTarget(self, action: #selector(fuck(sender:)), for: .touchDown)
        furniture2.addTarget(self, action: #selector(fuckme(sender:)), for: .touchUpInside)
        furniture2.addTarget(self, action: #selector(realfuckme(sender:)), for: .touchDownRepeat)
        furniture2.tag = 2
        
        detail1.backgroundColor = UIColor.detail
        detail1.frame = CGRect(x:0, y:275, width: 207, height: 400)

        detail2.backgroundColor = UIColor.detail
        detail2.frame = CGRect(x:207, y:275, width: 207, height: 400)
        
        trash.image = UIImage(imageLiteralResourceName: "trash.png")
        trash.frame = CGRect(x:310, y:10, width: 80, height: 80)
        trash.backgroundColor = UIColor.clear
        
        
        sceneView.addSubview(detail1)
        sceneView.addSubview(detail2)
        sceneView.addSubview(trash)
        
        detail1.isHidden = true
        detail2.isHidden = true
        
        info1.frame = CGRect(x:0,y:0,width:207,height:400)
        info2.frame = CGRect(x:0, y:0, width:207, height:400)
        info1.backgroundColor = UIColor.detail
        info2.backgroundColor = UIColor.detail
        info1.text = "This is 1.\n This should be the second line"
        info2.text = "This is 2.\n This should be the second line"
        detail1.addSubview(info1)
        detail2.addSubview(info2)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    
        // self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints,ARSCNDebugOptions.showWorldOrigin]
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        self.sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        
        guard currentBuffer == nil, case .normal = frame.camera.trackingState else {
            return
        }
        currentBuffer = frame.capturedImage


    
        self.cameraInverse = frame.camera.transform.inverse

        startDetection()
    }
    
    
    // MARK: - Setup
    func setUpModel() {
        guard let depthModel = try? VNCoreMLModel(for: FCRN().model) else {
            fatalError("Could not load depth model.")
        }
        guard let segmentationModel = try? VNCoreMLModel(for: DeepLabV3().model) else {
            fatalError("Could not load segmentation model.")
        }
        print("Finished Loading Models")

        let depthRequest = VNCoreMLRequest(model: depthModel, completionHandler: depthCompleteHandler)
        depthRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        let segmentationRequest = VNCoreMLRequest(model: segmentationModel, completionHandler: segmentationCompleteHandler)
        segmentationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        self.visionRequestSeg = [segmentationRequest]
        self.visionRequestDepth = [depthRequest]
    }
    
    
    // New method to create a bunch of bilboards
    func bilbordsCreate() {
        // 我觉得应该每个单独建 不然一改全改了
        for i in 0...(max_segmentation_count - 1) {

            let maskMaterial = SCNMaterial()
            maskMaterial.diffuse.contents = UIColor(white: 1, alpha: 0)
            maskMaterial.colorBufferWriteMask = .alpha
            let rectangleDepth = SCNPlane(width: 0.0464, height: 0.058)
            rectangleDepth.materials = [maskMaterial]
            let maskNode = SCNNode(geometry: rectangleDepth)
            maskNode.eulerAngles = SCNVector3Make(0, 0, 0)
            maskNode.position = SCNVector3Make(0, 0, -0.05)
            maskNode.renderingOrder = 0
            maskNode.name = "mask" + String(i)
            sceneView.pointOfView?.presentation.addChildNode(maskNode)
            maskNodes[i] = MYNode.init(node:maskNode) 
        }
    }
    
    func isVirtualID(id: Int32) -> Bool {
        return id >= 40;
    }
    
    // Update the bilbords based on the sorted array
    func bilbordUpdate() {
        print("#########update bilborads############")
        let allTuples: [(Int32, Double)] = depthSort(segMap: MLMultiArray.getSegmentDepthTuple(segmentationMap: segmentationMap!, depthMap: depthMap!)!)
        
        print(allTuples)
        var segIDArray: [Int32] = []
        var renderingOrder = -100
        var maskIndex = 0
        for tuple in allTuples {
            if (!isVirtualID(id: tuple.0)) {
                segIDArray.append(tuple.0)
            } else {
                if let image = MLMultiArray.buildFromSegID(segmentationMap: segmentationMap!, segIDArray: segIDArray){
                    modifyMaskNode(index: maskIndex, renderingOrder: renderingOrder, image: image)
                    segIDArray = []
                    renderingOrder += 1
                    maskIndex += 1
                }
                if let modelNode = registeredModels[tuple.0]
                {
                    modelNode.myNode.renderingOrder = renderingOrder
                    renderingOrder += 1
                }
                // TODO: reset all mask nodes
            }
        }
    
    }
    
    // helper function to modify a single mask node
    func modifyMaskNode(index: Int, renderingOrder: Int, image: CGImage) {
        maskNodes[index].myNode.geometry?.materials[0].diffuse.contents = image
        maskNodes[index].myNode.renderingOrder = renderingOrder
    }
    
    func depthSort(segMap:[(Int32, Double)]) -> [(Int32, Double)]{
 
        //virtual obejects
        
        var modelTuples : [(Int32, Double)] = []
        for i in registeredModels.keys{
            
            if let depth = registeredModels[i]?.myDepth {
               let a : (Int32, Double) = (i, depth)
               modelTuples.append(a)
            }
        
        }
        var allTuples = segMap + modelTuples
        allTuples.sort(by: {$0.1 < $1.1})
        return allTuples
    }
    
    
    @objc func addShipToSceneView(withGestureRecognizer recognizer: UIGestureRecognizer) {
        let tapLocation = recognizer.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(tapLocation, types: .existingPlaneUsingExtent)
        
        guard let hitTestResult = hitTestResults.first else { return }
        let translation = hitTestResult.worldTransform.translation
        let x = translation.x
        let y = translation.y
        let z = translation.z
        let shipScene = SCNScene(named: "3D Objects/chair1/chair1.scn")
        guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
                else {debugPrint("NO MODEL!")
                    return }

        shipNode.worldPosition = SCNVector3(x,y,z)
        shipNode.renderingOrder = -1
        shipNode.name = "furniture" + String(id)
        sceneView.scene.rootNode.addChildNode(shipNode)
        let myShipNode = MYNode.init(node: shipNode, id: id, depth: -1.0)
        setNodeDepth(node: myShipNode)
        registeredModels[id] = myShipNode
        id += 1

    }
        
    @objc func dragModelInSceneView(panGesture: UIPanGestureRecognizer) {
            let location = panGesture.location(in: sceneView)
            switch panGesture.state {
                case .began:
                    //debugPrint("###############Dragging##############")
                    guard let hitNodeResult = sceneView.hitTest(location, options: [SCNHitTestOption.searchMode: 1]).first(where: {($0.node.name?.hasPrefix("furniture") ?? false)}) else{return}
                    draggingNode = hitNodeResult.node
                    dragId = draggingNode.getID()
                    // name furniture[id]
                    //debugPrint(dragId)
                    //debugPrint(draggingNode)
                    //debugPrint(registeredModels[dragId]?.myNode)
                
                case .changed:
                    let location = panGesture.location(in: sceneView)
                    if (location.x > 310 && location.y < 90){
                        draggingNode.removeFromParentNode()
                        return
                    }
                    guard let hitTestResult = sceneView.hitTest(location,types: .existingPlaneUsingExtent).first else {return}
                    let translation = hitTestResult.worldTransform.translation
                    let x = translation.x
                    let y = translation.y
                    let z = translation.z
                    draggingNode.position = SCNVector3(x,y,z)
                    
                case .ended:
                    if let model = registeredModels[dragId]
                    {
                        setNodeDepth(node: model)
                    }
                    //debugPrint(registeredModels[dragId]?.myDepth)
                    dragId = nil
                    draggingNode = nil
                //debugPrint("###############Dragging Finish##############")
                default:
                    return
            }
        }
    
    func setNodeDepth(node: MYNode) {
        //debugPrint("---setting depth------")
        let position = node.myNode.position
        let virtualCoord = simd_float4(position.x, position.y, position.z, 1)
        if let inverse = self.cameraInverse
        {
            
            let result = inverse * virtualCoord
            node.myDepth = Double(-result.z)
            //debugPrint(node.myDepth)
            //debugPrint("---setting finish------")
        }
        else{
            debugPrint("---setting fail------")
            //debugPrint(cameraInverse)
        }

        
    }
    
    func addTapGestureToSceneView() {
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(ViewController.addShipToSceneView(withGestureRecognizer:)))
        sceneView.addGestureRecognizer(tapGestureRecognizer)
        debugPrint("Knock Knock")
    }

    func addPanGestureToSceneView() {
        let panGestureRecognizer = UIPanGestureRecognizer(target: self, action:
            #selector(ViewController.dragModelInSceneView(panGesture:)))
        sceneView.addGestureRecognizer(panGestureRecognizer)
        debugPrint("Pan is Ready to Burn")
    }
    
    @IBAction func showFurnitures(_ sender: UIButton) {
        var image1: UIImage?
        var image2: UIImage?
        furniture1.removeFromSuperview()
        furniture2.removeFromSuperview()
        if (typeButton == sender.tag) {
//            self.furniture1.isHidden = true
//            hStack.removeArrangedSubview(furniture1)
//            self.furniture2.isHidden = true
//            furniture2.removeFromSuperview()
            typeButton = 5
        }
        else {
            // Set image first
            switch sender.tag {
            case 0:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/table1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/table2.png")
            case 2:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/bed1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/bed2.png")
            case 4:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/cup1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/cup2.png")
            case 6:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/chair1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/chair2.png")
            case 8:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/sofa1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/sofa2.png")
            default:
                return
            }
            
            typeButton = sender.tag
            infoindex = sender.tag
            // Set up stack and buttons
            hStack = {
                let stack = UIStackView()
                stack.frame = CGRect(x:0, y:675, width:414, height: 131)
                stack.axis = NSLayoutConstraint.Axis.horizontal
                stack.distribution = .fillEqually
                stack.spacing = 0
                return stack
            }()
            
            
            furniture1.setImage(image1, for: .normal)
            furniture2.setImage(image2, for: .normal)
            
            // Add furniture to stack and stack to subview
//            hStack.addArrangedSubview(furniture1)
//            hStack.addArrangedSubview(furniture2)
            hStack.addArrangedSubview(furniture1)
            hStack.addArrangedSubview(furniture2)
            
            sceneView.addSubview(hStack)
//            hStack.addBackground(color: UIColor.white)
        }
    }
    
    @objc func buttonAction(sender: UIButton!) {
        if (maskMaterial.colorBufferWriteMask == .all) {
            maskMaterial.colorBufferWriteMask = .alpha
            sender.setTitle("Show mask", for: .normal)
            return
        }
        maskMaterial.colorBufferWriteMask = .all
        sender.setTitle("Hide mask", for: .normal)
    }
    
    @objc func fuck(sender : UIButton) {
        if (sender.tag == 1) {
            detail1.isHidden = false
            info1.text = infoarray[infoindex]
        }
        else if (sender.tag == 2) {
            detail2.isHidden = false
            info2.text = infoarray[infoindex+1]
        }
    }
    
    @objc func fuckme(sender : UIButton) {
        if (sender.tag == 1) {
            detail1.isHidden = true
        }
        else if (sender.tag == 2) {
            detail2.isHidden = true
        }
//        detail1.removeFromSuperview()
    }
    
    @objc func realfuckme(sender : UIButton){
            debugPrint("AAAASSSS")
            
//            var a = models[infoindex+sender.tag-1]
//            debugPrint(a)
            let shipScene = SCNScene(named: models[infoindex+sender.tag-1])
            shipScene?.rootNode.scale = SCNVector3(0.1,0.1,0.1)
            guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
            else {debugPrint("NO MODEL!")
                return }
            shipNode.eulerAngles = SCNVector3(90,0,0)
            shipNode.position = SCNVector3(0,0,-3)
            shipNode.renderingOrder = -1

//            sceneView.pointOfView?.addChildNode(shipNode)
            sceneView.scene.rootNode.addChildNode(shipNode)
        }
}

extension UIStackView{
    func addBackground(color: UIColor) {
        let subview = UIView(frame: bounds)
        subview.backgroundColor = color
        subview.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        insertSubview(subview, at: 0)
    }
}

extension ViewController {
    // MARK: - CoreML Prediction
    private func startDetection() {
            // To avoid force unwrap in VNImageRequestHandler
            guard let buffer = currentBuffer else { return }
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)

            //Run in background thread
            visionQueue.async {
                do {
                    try imageRequestHandler.perform(self.visionRequestSeg)
                } catch {
                    print(error)
                }
            }
        }
    
    // MARK: - Request Handler
    func segmentationCompleteHandler(request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        
        if let observations = request.results as? [VNCoreMLFeatureValueObservation] {
            if let segmentationOutput = observations.first?.featureValue {
                guard let array = segmentationOutput.multiArrayValue else {
                    return
                }
                //print(array.shape)
                segmentationMap = array
                
                guard let buffer = currentBuffer else { return }
                let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)
                visionQueue.async {
                    do {
                        try imageRequestHandler.perform(self.visionRequestDepth)
                    } catch {
                        print(error)
                    }
                }
            }
        }
    }
    
    func depthCompleteHandler(request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        if let observations = request.results as? [VNCoreMLFeatureValueObservation] {
            if let depthOutput = observations.first?.featureValue {
                guard let array = depthOutput.multiArrayValue else {
                    return
                }
                guard let newArray = array.to2DArray() else {
                    return
                }
                //print(newArray.shape)
                depthMap = newArray
                
//                let CGimage = MLMultiArray.buildFromSegmentationDepth(segmentationMap: self.segmentationMap!, depthMap: self.depthMap!, threshold: Double(self.depthThreshold) + 0.3)

                self.bilbordUpdate()

                self.ReleaseBuffer()
            }
        }
    }
    
    private func ReleaseBuffer() {
        // The resulting image (mask) is available as observation.pixelBuffer
        // Release currentBuffer when finished to allow processing next frame
        self.currentBuffer = nil
        self.segmentationMap = nil
        self.depthMap = nil
    }

    
    // MARK: - Renderer
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        // 1
        guard let planeAnchor = anchor as? ARPlaneAnchor else { return }
        
        // 2
        let width = CGFloat(planeAnchor.extent.x)
        let height = CGFloat(planeAnchor.extent.z)
        let plane = SCNPlane(width: width, height: height)
        
        // 3
        
        plane.materials.first?.diffuse.contents = UIColor.idkColor
//        plane.materials.first?.isDoubleSided = true
        
        // 4
        let planeNode = SCNNode(geometry: plane)
        planeNode.renderingOrder = -1
        
        // 5
        let x = CGFloat(planeAnchor.center.x)
        let y = CGFloat(planeAnchor.center.y)
        let z = CGFloat(planeAnchor.center.z)
        planeNode.position = SCNVector3(x,y,z)
        planeNode.eulerAngles.x = -.pi / 2
        
        // 6
        node.addChildNode(planeNode)
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        // 1
        guard let planeAnchor = anchor as?  ARPlaneAnchor,
            let planeNode = node.childNodes.first,
            let plane = planeNode.geometry as? SCNPlane
            else { return }
        
        // 2
        let width = CGFloat(planeAnchor.extent.x)
        let height = CGFloat(planeAnchor.extent.z)
        plane.width = width
        plane.height = height
        
        // 3
        let x = CGFloat(planeAnchor.center.x)
        let y = CGFloat(planeAnchor.center.y)
        let z = CGFloat(planeAnchor.center.z)
        planeNode.position = SCNVector3(x, y, z)
    }
}

extension SCNNode {
    
    func getID() -> Int32{
        var id:Int32  = -1
        if let name = self.name{
            let idString = String(name.suffix(2))
            print("jinglaile")
            print(idString)
            id = Int32(idString) ?? -1
        }
        return id
        
    }
}

extension float4x4 {
    var translation: float3 {
        let translation = self.columns.3
        return float3 (translation.x, translation.y, translation.z)
    }
}

extension UIColor {
    open class var transparentLightBlue: UIColor {
        return UIColor(red: 90/255, green: 200/255, blue: 250/255, alpha: 0.60)
    }
    
    open class var idkColor: UIColor {
        return UIColor(white: 0.0, alpha: 0.7)
    }
    
    open class var detail: UIColor {
        return UIColor(red: 255/255, green: 255/255, blue: 255/255, alpha: 0.30 )
    }
}




