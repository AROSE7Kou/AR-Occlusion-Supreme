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
    let objectDectectionModel = YOLOv3Tiny()
    //let depthDetectionModel = DeepLabV3FP16()
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
    var draggingNode: SCNNode?
    
    // MARK: - Button Controller
    var typeButton = -1
    var hStack: UIStackView! = UIStackView()
    var furniture1:UIButton! = UIButton()
    var furniture2:UIButton! = UIButton()
    var trash:UIImageView! = UIImageView()
    var detail1: UIView! = UIView()
    var detail2: UIView! = UIView()
    var info1: UITextView! = UITextView()
    var info2: UITextView! = UITextView()
    var infoindex = 0
    let models:[String] = ["3D Objects/table0.scn", "3D Objects/table1.scn", "3D Objects/desk0.scn", "3D Objects/desk1.scn", "3D Objects/cup0.scn", "3D Objects/cup1.scn", "3D Objects/chair0.scn", "3D Objects/ship.scn", "3D Objects/sofa0.scn", "3D Objects/sofa1.scn"]
//    @objc func initialConvert(){
//    }
    // MARK: - TableView Data
    var predictions: [VNRecognizedObjectObservation] = []
    
    // MARK - Performance Measurement Property
    //private let 👨‍🔧 = 📏()
    
    let configuration = ARWorldTrackingConfiguration()
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        let vaseScene = SCNScene(named:"IronMan/IronMan.scn")
        guard let ironmannode =  vaseScene?.rootNode else { return }
        ironmannode.position = SCNVector3(0,-0.3,-1)
        
        
//        guard let modelScene = SCNScene(mdlAsset: mdlAsset),
//            let nodeModel =  modelScene.rootNode.childNode(
//               withName: "vase", recursively: true)
//        else{
//            print("fails")
//            return}

        
        sceneView.delegate = self
//
//         Show statistics such as fps and timing information
        //sceneView.showsStatistics = true
//
//         Create a new scene
       //let scene = SCNScene()
//
//         Set the scene to the view
        //sceneView.scene = scene
        sceneView.scene.rootNode.addChildNode(ironmannode)
        sceneView.scene.rootNode.addChildNode(planesRootNode)

//
//         Enable Default Lighting - makes the 3D text a bit poppier.
        sceneView.autoenablesDefaultLighting = true
        addTapGestureToSceneView()
        addPanGestureToSceneView()
        // setup the model
        setUpModel()
        
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
        info1.tag = 0
        info2.tag = 1
        
        detail1.addSubview(info1)
        detail2.addSubview(info2)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    
        // self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints,ARSCNDebugOptions.showWorldOrigin]
        self.sceneView.session.run(configuration)
        setUpSceneView()

        self.loopCoreMLUpdate()
        //self.videoCapture.start()
        
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        //self.videoCapture.stop()
        sceneView.session.pause()
    }
    
    func setUpSceneView() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        
        sceneView.session.run(configuration)
        
        sceneView.delegate = self
        // sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
    }
    
    func loopCoreMLUpdate() {
        // Continuously run CoreML whenever it's ready. (Preventing 'hiccups' in Frame Rate)
        let delay : Double = 0.2
        let time = DispatchTime.now() + delay
        DispatchQueue.main.asyncAfter(deadline:time) {
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
    
    @objc func addShipToSceneView(withGestureRecognizer recognizer: UIGestureRecognizer) {
        let tapLocation = recognizer.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(tapLocation, types: .existingPlaneUsingExtent)
        
        //debugPrint("TAPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")

                guard let hitTestResult = hitTestResults.first else { return }
                let translation = hitTestResult.worldTransform.translation
                let x = translation.x
                let y = translation.y
                let z = translation.z
                let shipScene = SCNScene(named: models[infoindex])
                guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
                    else {debugPrint("NO MODEL!")
                        return }
                shipNode.position = SCNVector3(x,y,z)
                shipNode.renderingOrder = -200

                sceneView.scene.rootNode.addChildNode(shipNode)
        
                debugPrint("Fuck UUUUUUUUUUUUU")
            }
        
    @objc func dragModelInSceneView(panGesture: UIPanGestureRecognizer) {
            let location = panGesture.location(in: sceneView)
            switch panGesture.state {
                case .began:
                    guard let hitNodeResult = sceneView.hitTest(location, options: nil).first else{return}
                    draggingNode = hitNodeResult.node
                    if (draggingNode?.name != "furniture") {
                        draggingNode = nil
                        debugPrint("ABCDE")
                        return
                    }
                    debugPrint("AHHHHHHHHH")
                case .changed:
                    let location = panGesture.location(in: sceneView)
                    if ((location.x > 350) && (location.y<80)){
                        draggingNode?.removeFromParentNode()
                        return
                    }
                    //debugPrint(location)
                    
                    guard let hitTestResult = sceneView.hitTest(location,types: .existingPlaneUsingExtent).first else {
                        debugPrint("No Plane")
                        return
                    }
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
            //debugPrint("Knock Knock")
        }
    
        func addPanGestureToSceneView() {
            let panGestureRecognizer = UIPanGestureRecognizer(target: self, action:
                #selector(ViewController.dragModelInSceneView(panGesture:)))
            sceneView.addGestureRecognizer(panGestureRecognizer)
            //debugPrint("Pan is Ready to Burn")
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
            infoindex = sender.tag
            //var text = infoarray[infoindex]
        
            // Set image first
            switch sender.tag {
            case 0:
                image1 = UIImage(imageLiteralResourceName: "furniture pics/table1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/table2.png")
            case 2:
                
                image1 = UIImage(imageLiteralResourceName: "furniture pics/bed1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/bed2.png")
            case 4:
                //infoindex = 4
                image1 = UIImage(imageLiteralResourceName: "furniture pics/cup1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/cup2.png")
            case 6:
                //infoindex = 6
                image1 = UIImage(imageLiteralResourceName: "furniture pics/chair1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/chair2.png")
            case 8:
                //infoindex = 8
                image1 = UIImage(imageLiteralResourceName: "furniture pics/sofa1.png")
                image2 = UIImage(imageLiteralResourceName: "furniture pics/sofa2.png")
            default:
                return
            }
            
            typeButton = sender.tag
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
    
    var infoarray = infospill()
    
    @objc func Modelpop(sender : UIButton){
        
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
        var scalex = 0.0
        var scaley = 0.0
        var scalez = 0.0
        var index = infoindex+sender.tag-1
        var a = models[index]
        debugPrint(a)
        let shipScene = SCNScene(named: models[infoindex+sender.tag-1])
        guard let shipNode = shipScene?.rootNode.childNode(withName: "furniture", recursively: false)
        else {debugPrint("NO MODEL!")
            return }
        shipNode.position = SCNVector3(0,0,-1)
        switch index {
            case 0:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 1:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 2:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 3:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 4:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 5:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 6:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 7:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 8:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            case 9:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
            default:
                scalex = 0.0007
                scaley = 0.0007
                scalez = 0.0007
        }
        shipNode.scale = SCNVector3(scalex,scaley,scalez)
        shipNode.renderingOrder = -200

//        sceneView.pointOfView?.addChildNode(shipNode)
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
                    //print("the height and width:")
                    //print(width)
                    //print(height)
                    
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
                {                planesRootNode.replaceChildNode(planesRootNode.childNodes[currentPlaneCount], with: node)
                    currentPlaneCount += 1
                    currentPlaneCount %= maxPlanesCount
                }
                else
                {
                    planesRootNode.addChildNode(node)
                }

            }
            
            guard let currentPosition = sceneView.pointOfView?.position else {return}
            for nodes in planesRootNode.childNodes {
                nodes.look(at: currentPosition)
            }
        }
    }
    
    func createPlane(rect : CGRect, coordinate: SCNVector3, planeWidth : CGFloat, planeHeight : CGFloat) -> SCNNode
    {
        let plane = SCNPlane(width: planeWidth, height: planeHeight)
        plane.cornerRadius = 0.05
        let planeNode = SCNNode(geometry: plane)
        planeNode.geometry?.firstMaterial?.isDoubleSided = true
        planeNode.geometry?.firstMaterial?.colorBufferWriteMask = .alpha
        planeNode.geometry?.firstMaterial?.writesToDepthBuffer = true
        planeNode.geometry?.firstMaterial?.readsFromDepthBuffer = true
        planeNode.renderingOrder = -300
        // Make the plane visible
        planeNode.geometry?.firstMaterial = maskMaterial()
//        
//        planeNode.physicsBody = SCNPhysicsBody(type: .static,
//                                              shape: nil)
        planeNode.position = coordinate
        return planeNode
    }
    
    func maskMaterial() -> SCNMaterial {
        let maskMaterial = SCNMaterial()
        maskMaterial.diffuse.contents = UIColor.transparentLightBlue
        
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
        planeNode.renderingOrder = -300
        
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
        return UIColor(red: 90/255, green: 200/255, blue: 250/255, alpha: 0.30)
    }
    
    open class var idkColor: UIColor {
        return UIColor(white: 0.0, alpha: 0.8)
    }
    
    open class var detail: UIColor {
        return UIColor(red: 255/255, green: 255/255, blue: 255/255, alpha: 0.30 )
    }
}



