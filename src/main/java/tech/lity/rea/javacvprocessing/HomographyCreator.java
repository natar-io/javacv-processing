/*
 * Part of the PapARt project - https://project.inria.fr/papart/
 *
 * Copyright (C) 2018 RealtiyTech
 * Copyright (C) 2014-2016 Inria
 * Copyright (C) 2011-2013 Bordeaux University
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, version 2.1.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with this library; If not, see
 * <http://www.gnu.org/licenses/>.
 */
package tech.lity.rea.javacvprocessing;

import java.util.List;
import static org.bytedeco.javacpp.opencv_calib3d.cvFindHomography;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.bytedeco.javacpp.opencv_core.cvCreateMat;
import org.bytedeco.javacv.JavaCV;
import processing.core.PMatrix3D;
import processing.core.PVector;
import toxi.geom.Matrix4x4;

/**
 *
 * @author Jeremy Laviole laviole@rea.lity.tech
 */
public class HomographyCreator {

    private opencv_core.CvMat cvMat;
    private Matrix4x4 mat;
    private PMatrix3D pmatrix;

    private opencv_core.CvMat srcPoints;
    private opencv_core.CvMat dstPoints;

    private final int srcDim;
    private final int dstDim;
    private final int nbPoints;


    private int currentPoint = 0;

    public HomographyCreator(int srcDim, int dstDim, int nbPoints) {
        this.srcDim = srcDim;
        this.dstDim = dstDim;
        this.nbPoints = nbPoints;
        init();
    }

    private void init() {
        currentPoint = 0;
        srcPoints = cvCreateMat(srcDim, nbPoints, CV_32FC1);
        dstPoints = cvCreateMat(dstDim, nbPoints, CV_32FC1);
        cvMat = cvCreateMat(3, 3, CV_32FC1);
    }
    
    public boolean addPoint(PVector src, PVector dst) {
        addPointCvMat(srcPoints, src);
        addPointCvMat(dstPoints, dst);
        currentPoint++;
        return checkAndComputeHomography();
    }

    private void addPointCvMat(opencv_core.CvMat points, PVector point) {
        points.put(currentPoint, point.x);
        points.put(currentPoint + nbPoints, point.y);
        if (points == srcPoints && srcDim == 3
                || points == dstPoints && dstDim == 3) {
            points.put(currentPoint + (nbPoints * 2), point.z);
        }
    }

    private boolean checkAndComputeHomography() {
        if (currentPoint == nbPoints) {
            createHomography();
            return true;
        }
        return false;
    }

    private void createHomography() {
        cvFindHomography(srcPoints, dstPoints, cvMat);
        currentPoint = 0;

        if (srcDim == dstDim && srcDim == 2) {
            mat = new Matrix4x4(cvMat.get(0), cvMat.get(1), 0, cvMat.get(2),
                    cvMat.get(3), cvMat.get(4), 0, cvMat.get(5),
                    0, 0, 1, 0,
                    0, 0, 0, 1);
        } else {
            mat = new Matrix4x4(cvMat.get(0), cvMat.get(1), cvMat.get(2), 0,
                    cvMat.get(3), cvMat.get(4), cvMat.get(5), 0,
                    cvMat.get(6), cvMat.get(7), cvMat.get(8), 0,
                    0, 0, 0, 1);
        }
        this.pmatrix = new PMatrix3D(
                (float) mat.matrix[0][0],
                (float) mat.matrix[0][1],
                (float) mat.matrix[0][2],
                (float) mat.matrix[0][3],
                (float) mat.matrix[1][0],
                (float) mat.matrix[1][1],
                (float) mat.matrix[1][2],
                (float) mat.matrix[1][3],
                (float) mat.matrix[2][0],
                (float) mat.matrix[2][1],
                (float) mat.matrix[2][2],
                (float) mat.matrix[2][3],
                (float) mat.matrix[3][0],
                (float) mat.matrix[3][1],
                (float) mat.matrix[3][2],
                (float) mat.matrix[3][3]);

//        homographyCalibrationOutput.setMatrix(pmatrix);
    }

    public PMatrix3D getMatrix(){
        return this.pmatrix;
    }
    
//    public boolean isComputed() {
//        return this.homographyCalibrationOutput.isValid();
//    }
//
//    public HomographyCalibration getHomography() {
//        assert (isComputed());
//        return this.homographyCalibrationOutput;
//    }

    @Override
    public String toString() {
        return this.mat.toString();
    }
    
    
    public static opencv_core.CvMat createHomography(List<PVector> in, List<PVector> out) {
        opencv_core.CvMat srcPoints;
        opencv_core.CvMat dstPoints;
        int nbPoints = in.size();
        opencv_core.CvMat homography;

        // TODO: no create map
        srcPoints = cvCreateMat(2, in.size(), opencv_core.CV_32FC1);
        dstPoints = cvCreateMat(2, in.size(), opencv_core.CV_32FC1);
        homography = cvCreateMat(3, 3, opencv_core.CV_32FC1);

        if (in.size() == 4 && out.size() == 4) {

            double[] src = new double[8];
            double[] dst = new double[8];
//            CvMat map = CvMat.create(3, 3);
            for (int i = 0; i < 4; i++) {
                src[i * 2] = in.get(i).x;
                src[i * 2 + 1] = in.get(i).y;
                dst[i * 2] = out.get(i).x;
                dst[i * 2 + 1] = out.get(i).y;
            }
//            System.out.println("JavaCV perspective...");
            JavaCV.getPerspectiveTransform(src, dst, homography);
            return homography;
        }

        for (int i = 0; i < in.size(); i++) {
            srcPoints.put(i, in.get(i).x);
            srcPoints.put(i + nbPoints, in.get(i).y);
            dstPoints.put(i, out.get(i).x);
            dstPoints.put(i + nbPoints, out.get(i).y);
        }
        cvFindHomography(srcPoints, dstPoints, homography);

//        opencv_imgproc.cvGetPerspectiveTransform(cpd, cpd1, srcPoints) //       It is better to use : GetPerspectiveTransform
        return homography;
    }

    public static opencv_core.CvMat createHomography(PVector[] in, PVector[] out) {
        opencv_core.CvMat srcPoints;
        opencv_core.CvMat dstPoints;
        int nbPoints = in.length;
        opencv_core.CvMat homography;
        // TODO: no create map
        srcPoints = cvCreateMat(2, in.length, opencv_core.CV_32FC1);
        dstPoints = cvCreateMat(2, in.length, opencv_core.CV_32FC1);
        homography = cvCreateMat(3, 3, opencv_core.CV_32FC1);
        for (int i = 0; i < in.length; i++) {
            srcPoints.put(i, in[i].x);
            srcPoints.put(i + nbPoints, in[i].y);
            dstPoints.put(i, out[i].x);
            dstPoints.put(i + nbPoints, out[i].y);
        }
        cvFindHomography(srcPoints, dstPoints, homography);
        //       It is better to use : GetPerspectiveTransform
        return homography;
    }


}
