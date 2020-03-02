//
//  ViewController.swift
//  SSDMobileNet-CoreML
//
//  Created by GwakDoyoung on 01/02/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit
import ARKit
import Fritz
import RealityKit
import SceneKit.ModelIO
import Vision
import CoreMedia


class ViewController: UIViewController , ARSCNViewDelegate, ARSessionDelegate{

    // MARK: - UI Properties
    @IBOutlet weak var boxesView: DrawingBoundingBoxView!
    @IBOutlet weak var sceneView: ARSCNView!
    let deepLabModel = DeepLabV3()
    let objectDectectionModel = YOLOv3Int8LUT()
    private lazy var fritzModel = FritzVisionPeopleSegmentationModelFast()
    let visionQueue = DispatchQueue (label: "com.vision.ARML.visionqueue")

    var planeNode = SCNNode()
    let semaphore = DispatchSemaphore(value: 1)

    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var segRequest: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var segModel: VNCoreMLModel?
    var isInferencing = false
    var currentFrameBuffer: CVPixelBuffer?
    var currentFrameImage = CIImage()
    var planesNodes: [SCNNode] = []
    
    // MARK: - AV Property
    var videoCapture: VideoCapture!
    var planesRootNode = SCNNode()
    
    var maskMaterialImage : CGImage?
    
    // MARK: - Current Camera Orientation
    var eularAngle_x : simd_float4x4?
    var eularAngle_y : simd_float4x4?
    var eularAngle_z : simd_float4x4?
    // MARK: - TableView Data
    var predictions: [VNRecognizedObjectObservation] = []

    
    let configuration = ARWorldTrackingConfiguration()
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        let vaseScene = SCNScene(named:"IronMan/IronMan.scn")
        guard let node =  vaseScene?.rootNode else {print("no node"); return }

        
        sceneView.delegate = self
        sceneView.session.delegate = self


        let plane = SCNPlane(width: 0.424, height: 0.832)
        self.planeNode = SCNNode(geometry: plane)
        self.planeNode.geometry?.firstMaterial?.colorBufferWriteMask = .all
        self.planeNode.renderingOrder = -100
        self.planeNode.position = SCNVector3Make(0, 0, -1)
        sceneView.scene.rootNode.addChildNode(node)
        sceneView.scene.rootNode.addChildNode(planesRootNode)
        sceneView.pointOfView?.presentation.addChildNode(self.planeNode)
        //sceneView.pointOfView?.presentation.addChildNode(planesRootNode)

        setUpModel()
//        setUpCamera()
        FritzCore.configure()
        
    }
    
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
//        print(configuration.videoFormat.imageResolution)
//        configuration.videoFormat.imageResolution
        self.sceneView.session.run(configuration)

        
    }
//
//    override func viewWillDisappear(_ animated: Bool) {
//        super.viewWillDisappear(animated)
//        sceneView.session.pause()
//    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
//        self.semaphore.wait()
//        guard currentFrameBuffer == nil, case .normal = frame.camera.trackingState else {
//            print(currentFrameBuffer == nil)
////            print("no buffer");
//            return
//        }
        currentFrameBuffer = frame.capturedImage
        self.updateCoreML(frame: frame)
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
        
        if let segmentationModel = try? VNCoreMLModel(for: deepLabModel.model){
            self.segModel = segmentationModel
            segRequest = VNCoreMLRequest(model: segmentationModel, completionHandler:  segmentationComplete)
            segRequest?.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        }else{
            fatalError("cao")
        }
    }
    

    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = sceneView.bounds
    }
}


extension ViewController {
    func predictUsingVision(imageFromArkitScene: CVPixelBuffer) {
        guard let request = request else { print("no request"); fatalError() }
//        self.semaphore.wait()

//        print("before")
//        print(CVPixelBufferGetWidth(imageFromArkitScene))
//        print(CVPixelBufferGetHeight(imageFromArkitScene))
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: imageFromArkitScene, orientation: .leftMirrored)
        try? handler.perform([request])

    }
    
    
    func drawBox(bgRect: CGRect) {
        let bgView = UIView(frame: bgRect)
        bgView.layer.borderColor = UIColor.black.cgColor
        bgView.layer.borderWidth = 4
        bgView.backgroundColor = UIColor.clear
        self.sceneView.addSubview(bgView)
    }
    
    // MARK: - Post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let predictions = request.results as? [VNRecognizedObjectObservation] {
            DispatchQueue.main.async {
                self.predictions = predictions
                self.boxesView.subviews.forEach({ $0.removeFromSuperview() })
                self.sceneView.subviews.forEach({ $0.removeFromSuperview() })
                if self.predictions.count == 0{
                    return
                }
                let prediction = self.predictions[0]

                var rect: CGRect = self.boxesView.createLabelAndBox(prediction: prediction)
        //            let flip = CGAffineTransform(scaleX: 1, y: -1)
                let flip = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 104.867, y: -832)
                rect = rect.applying(flip)
        //            let smallerPiece = scaleBufferImage(input: self.currentFrameImage.cropped(to: rect))
                let smallerPiece = self.currentFrameImage.cropped(to: rect)
                if let cgmask = self.convertCIImageToCGImage(inputImage: smallerPiece)
                {
                    self.planeNode.geometry?.firstMaterial?.diffuse.contents = cgmask
                }
            }
                
//                if let segRequest = segRequest {
//                    let supImageRequestHandler = VNImageRequestHandler(ciImage: smallerPiece, options: [:])
//                    try? supImageRequestHandler.perform([segRequest])
//                    if let maskMaterialImage = self.maskMaterialImage {
//                        self.addPlane(rect: rect, maskCGImage: maskMaterialImage)
//                        print("maskmaterial###########")
//                    } else {print("fail 1 ########")
//                            self.addPlane(rect: rect)}
//
//                } else { print("fail 2 ##########")
//                         self.addPlane(rect: rect) }

        }
        self.isInferencing = false
//        } else {
//
//            self.isInferencing = false
//        }
        self.ReleaseBuffer()
//        self.semaphore.signal()
    }

    func segmentationComplete(request: VNRequest, error: Error?) {
            // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }

        
        if let observations = request.results as? [VNCoreMLFeatureValueObservation] {
            if let segmentationmap = observations.first?.featureValue {
                let array = segmentationmap.multiArrayValue!
                let CGimage = array.cgImage(min: 0, max: 1)
                self.maskMaterialImage = CGimage
            }
        }
    }
    
    
    private func addPlane(rect : CGRect, maskCGImage : CGImage) {

        let screenCentre : CGPoint = CGPoint(x: rect.midX, y: rect.midY)
        let planeorigin : CGPoint = rect.origin
        let rightbottom : CGPoint = CGPoint(x: rect.maxX, y: rect.minY)
        let centerTestResults : [ARHitTestResult] = sceneView.hitTest(screenCentre, types: [.featurePoint])
        let originTestResults : [ARHitTestResult] = sceneView.hitTest(planeorigin, types: [.featurePoint])
        let rightbotTestResults : [ARHitTestResult] = sceneView.hitTest(rightbottom, types: [.featurePoint])
        if let closestResult = centerTestResults.first {
            if let originResult = originTestResults.first{
                if let rightbotResult = rightbotTestResults.first{
                    
                    
                    let originTransform : matrix_float4x4 = originResult.worldTransform
                    let rightbotTransform : matrix_float4x4 = rightbotResult.worldTransform
                    let width : CGFloat = CGFloat(abs(rightbotTransform.columns.3.x - originTransform.columns.3.x))
                    let height : CGFloat = CGFloat(abs(originTransform.columns.3.y - rightbotTransform.columns.3.y))
                    
                    var transform : matrix_float4x4 = closestResult.worldTransform
                    transform.columns.3.z = (transform.columns.3.z + originTransform.columns.3.z + rightbotTransform.columns.3.z)/3.0
                    let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                    // Create 3D Text
                    let node : SCNNode = createPlane(rect: rect, coordinate: worldCoord, planeWidth: width, planeHeight: height, maskCGImage: maskCGImage)

                    let finalRotateTransform = simd_mul(transform,eularAngle_y!)
                        node.transform = SCNMatrix4(finalRotateTransform)
                    planesRootNode.addChildNode(node)
                }
            }

        }

    }
    
    private func addPlane(rect : CGRect){

        let screenCentre : CGPoint = CGPoint(x: rect.midX, y: rect.midY)
        let planeorigin : CGPoint = rect.origin
        let rightbottom : CGPoint = CGPoint(x: rect.maxX, y: rect.maxY)
        let centerTestResults : [ARHitTestResult] = sceneView.hitTest(screenCentre, types: [.featurePoint])
        let originTestResults : [ARHitTestResult] = sceneView.hitTest(planeorigin, types: [.featurePoint])
        let rightbotTestResults : [ARHitTestResult] = sceneView.hitTest(rightbottom, types: [.featurePoint])
        if let closestResult = centerTestResults.first {
            if let originResult = originTestResults.first{
                if let rightbotResult = rightbotTestResults.first{
                    
                    
                    let originTransform : matrix_float4x4 = originResult.worldTransform
                    let rightbotTransform : matrix_float4x4 = rightbotResult.worldTransform
                    let width : CGFloat = CGFloat(abs(rightbotTransform.columns.3.x - originTransform.columns.3.x))
                    let height : CGFloat = CGFloat(abs(originTransform.columns.3.y - rightbotTransform.columns.3.y))
                    
                    var transform : matrix_float4x4 = closestResult.worldTransform
                    transform.columns.3.z = (transform.columns.3.z + originTransform.columns.3.z + rightbotTransform.columns.3.z)/3.0
                    let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                    // Create 3D Text
                    let node : SCNNode = createPlane(rect: rect, coordinate: worldCoord, planeWidth: width, planeHeight: height)
                    
                    let finalRotateTransform = simd_mul(transform,eularAngle_y! )
                    node.transform = SCNMatrix4(finalRotateTransform)
                    //planesNodes.append(node)
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
    
        planeNode.geometry?.firstMaterial?.writesToDepthBuffer = true
        planeNode.geometry?.firstMaterial?.readsFromDepthBuffer = true
        planeNode.renderingOrder = -100
        planeNode.geometry?.firstMaterial?.diffuse.contents = UIColor.orange
        planeNode.position = coordinate
        return planeNode
    }
    
    func createPlane(rect : CGRect, coordinate: SCNVector3, planeWidth : CGFloat, planeHeight : CGFloat, maskCGImage: CGImage) -> SCNNode
    {
        let plane = SCNPlane(width: planeWidth, height: planeHeight)
        plane.cornerRadius = 0.005
        let planeNode = SCNNode(geometry: plane)
        planeNode.geometry?.firstMaterial?.diffuse.contents = maskCGImage
        planeNode.geometry?.firstMaterial?.colorBufferWriteMask = .all
        planeNode.renderingOrder = -100
        planeNode.position = coordinate
        return planeNode
    }
    
    func scaleBufferImage(input: CIImage) -> CIImage {
        let srcWidth = CGFloat(input.extent.width) // 1920
        let srcHeight = CGFloat(input.extent.height) // 1440
        var output = CIImage()
        let dstWidth: CGFloat = self.sceneView.bounds.size.height   //832
        let dstHeight: CGFloat = self.sceneView.bounds.size.width   //414
//        output = input.cropped(to: cropRec)
        
        let scaleX = dstWidth / srcWidth
        let scaleY = dstHeight / srcHeight

        let transform = CGAffineTransform.identity.scaledBy(x: scaleX, y: scaleY)
        output = input.transformed(by: transform)
        let oTransform = output.orientationTransform(for: .right)
        output = output.transformed(by: oTransform)
        return output
    }
    
    func cropBufferImage(input: CIImage) -> CIImage {
//        let orientation = UIApplication.shared.windows.first?.windowScene?.interfaceOrientation;
        let srcWidth = CGFloat(input.extent.width) // 1920
        let srcHeight = CGFloat(input.extent.height) // 1440
        var output = CIImage()
        let dstWidth: CGFloat = self.sceneView.bounds.size.height   //832
        let dstHeight: CGFloat = self.sceneView.bounds.size.width   //414
        let centerY = srcHeight/2
        let targetHeight = dstHeight * srcWidth / dstWidth
        let cropRec = CGRect(x: 0,y: centerY - targetHeight / 2, width: srcWidth, height: targetHeight)
        output = input.cropped(to: cropRec)
        return output
    }
    

    //MARK: - updatePredictionByARscene
    func updateCoreML(frame: ARFrame) {
        planesRootNode.removeFromParentNode();
        planesRootNode = SCNNode();
        sceneView.scene.rootNode.addChildNode(planesRootNode)
        guard let buffer = currentFrameBuffer else { print("no buffer"); return }
        currentFrameImage = cropBufferImage(input: CIImage.init(cvPixelBuffer: buffer))
        eularAngle_y = simd_float4x4(SCNMatrix4MakeRotation(sceneView.session.currentFrame!.camera.eulerAngles.y, 0, 1, 0))
        let scaledBuffer = currentFrameImage.pixelBufferFromImage()
        currentFrameImage = scaleBufferImage(input: currentFrameImage)

        self.predictUsingVision(imageFromArkitScene: scaledBuffer)
        
    }
    
    //MARK: - help funcs
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
            return cgImage
        }
        return nil
    }
    
    
    private func ReleaseBuffer() {
         // The resulting image (mask) is available as observation.pixelBuffer
         // Release currentBuffer when finished to allow processing next frame
        self.currentFrameBuffer = nil
        self.maskMaterialImage = nil
     }
    
}


extension CVPixelBuffer {
    func transformedImage(targetSize: CGSize, rotationAngle: CGFloat) -> CIImage? {
        let image = CIImage(cvPixelBuffer: self, options: [:])
        let scaleFactor = Float(targetSize.width) / Float(image.extent.width)
        return image.transformed(by: CGAffineTransform(rotationAngle: rotationAngle)).applyingFilter("CIBicubicScaleTransform", parameters: ["inputScale": scaleFactor])
    }
}





