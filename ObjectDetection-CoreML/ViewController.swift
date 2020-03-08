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

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {

    // MARK: - UI Properties
    @IBOutlet weak var sceneView: ARSCNView!
    
    var maskNode : SCNNode!
    var maskMaterial : SCNMaterial!
    
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
    var draggingNode: SCNNode?
    
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
    let models:[String] = ["3D Objects/table0.scn", "3D Objects/table1.scn", "3D Objects/desk0.scn", "3D Objects/desk1.scn", "3D Objects/cup0.scn", "3D Objects/cup1.scn", "3D Objects/chair1.scn", "3D Objects/ship.scn", "3D Objects/sofa0.scn", "3D Objects/sofa1.scn"]
    
    // MARK - Performance Measurement Property
    //private let 👨‍🔧 = 📏()
    
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set Delegate
        sceneView.delegate = self
        sceneView.session.delegate = self
        
        sceneView.showsStatistics = true
        
        let vaseScene = SCNScene(named:"IronMan/IronMan.scn")
//        let vaseScene = SCNScene(named:"3D Objects/chair1.scn")
        guard let ironManNode =  vaseScene?.rootNode else { return }
        ironManNode.name = "IronMan"
        ironManNode.worldPosition = SCNVector3(0, -0.3, -1)
//        ironManNode.scale = SCNVector3(0.1,0.1,0.1)
        
        sceneView.scene.rootNode.addChildNode(ironManNode)
        

//      Enable Default Lighting - makes the 3D text a bit poppier.
        sceneView.autoenablesDefaultLighting = true
        addTapGestureToSceneView()
        addPanGestureToSceneView()
        setUpModel()
        bilbordCreate()
        
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
            
            // Calculate depth
            guard let position = sceneView.scene.rootNode.childNode(withName: "furniture", recursively: true)?.position else{
                self.ReleaseBuffer()
                return}
            let virtualCoord = simd_float4(position.x, position.y, position.z, 1)
            let result = frame.camera.transform.inverse * virtualCoord
            depthThreshold = -result.z
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
    
    func bilbordCreate() {
        maskMaterial = SCNMaterial()
        maskMaterial.diffuse.contents = UIColor(white: 1, alpha: 0)
        maskMaterial.colorBufferWriteMask = .alpha
        
        let rectangleDepth = SCNPlane(width: 0.0464, height: 0.058)
        rectangleDepth.materials = [maskMaterial]
        
        maskNode = SCNNode(geometry: rectangleDepth)
        maskNode?.eulerAngles = SCNVector3Make(0, 0, 0)
        maskNode?.position = SCNVector3Make(0, 0, -0.05)
        maskNode.renderingOrder = -2
        maskNode.name = "mask"
        sceneView.pointOfView?.presentation.addChildNode(maskNode!)
    }
    
    @objc func addShipToSceneView(withGestureRecognizer recognizer: UIGestureRecognizer) {
        let tapLocation = recognizer.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(tapLocation, types: .existingPlaneUsingExtent)
        
        debugPrint("TAPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")

        guard let hitTestResult = hitTestResults.first else { return }
        let translation = hitTestResult.worldTransform.translation
        let x = translation.x
        let y = translation.y
        let z = translation.z
        let shipScene = SCNScene(named: "3D Objects/chair1.scn")
        shipScene?.rootNode.scale = SCNVector3(0.1,0.1,0.1)
        guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
            else {debugPrint("NO MODEL!")
                return }
//        guard let shipNode = shipScene?.rootNode else {return}
//        shipNode.scale = SCNVector3(0.1, 0.1, 0.1)
        shipNode.worldPosition = SCNVector3(x,y,z)
        shipNode.renderingOrder = -1

        sceneView.scene.rootNode.addChildNode(shipNode)

        debugPrint("Fuck UUUUUUUUUUUUU")
    }
        
    @objc func dragModelInSceneView(panGesture: UIPanGestureRecognizer) {
        let location = panGesture.location(in: sceneView)
        switch panGesture.state {
            case .began:
                debugPrint("BEGAN")
                guard let hitNodeResult = sceneView.hitTest(location, options: [SCNHitTestOption.searchMode: 1]).first(where: {$0.node.name == "furniture"}) else{return}
                draggingNode = hitNodeResult.node

                debugPrint("GOTCHA")
            
            case .changed:
                let location = panGesture.location(in: sceneView)
                if (location.x > 310 && location.y < 90){
                    draggingNode?.removeFromParentNode()
                    return
                }
                guard let hitTestResult = sceneView.hitTest(location,types: .existingPlaneUsingExtent).first else {return}
                let translation = hitTestResult.worldTransform.translation
                let x = translation.x
                let y = translation.y
                let z = translation.z
                draggingNode?.position = SCNVector3(x,y,z)
                
//                    draggingNode?.look(at: <#T##SCNVector3#>)
            case .ended:
                draggingNode = nil
            default:
                return
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
            case 1:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/bed1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/bed2.png")
            case 2:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/cup1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/cup2.png")
            case 3:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/chair1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/chair2.png")
            case 4:
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
            guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
            else {debugPrint("NO MODEL!")
                return }

            shipNode.position = SCNVector3(0,0,-3)
            shipNode.renderingOrder = -1

            sceneView.pointOfView?.addChildNode(shipNode)
    //        sceneView.scene.rootNode.addChildNode(shipNode)
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

//                let CGimage = array.cgMask(min: 0, max: 1)
//                DispatchQueue.main.async {
//                    self.maskMaterial.diffuse.contents = CGimage
//                }
//                self.ReleaseBuffer()
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
                
                //print(Double(self.depthThreshold) + 0.3)
                let CGimage = MLMultiArray.buildFromSegmentationDepth(segmentationMap: self.segmentationMap!, depthMap: self.depthMap!, threshold: Double(self.depthThreshold) + 0.3)
                
//                let CGimage = newArray.cgImage(min: 0.5, max: 6)
//                let CGimage = array.cgMaskFCRN(min: 0.5, max: 5, threshold: Double(self.depthThreshold) + 0.3)
                DispatchQueue.main.async {
                    self.maskMaterial.diffuse.contents = CGimage
                }
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
        return UIColor(white: 0.0, alpha: 0.3)
    }
    
    open class var detail: UIColor {
        return UIColor(red: 255/255, green: 255/255, blue: 255/255, alpha: 0.30 )
    }
}



