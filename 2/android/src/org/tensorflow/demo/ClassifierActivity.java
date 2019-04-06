/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.*;
import android.graphics.Bitmap.Config;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;

import java.util.List;
import java.util.Vector;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    protected static final boolean SAVE_PREVIEW_BITMAP = false;

    private ResultsView resultsView;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap inputBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private long lastProcessingTimeMs;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(800, 480);
    private static final Size CROP_SIZE = new Size(200, 300);
    private static final int INPUT_SIZE = 96;
    private static final float TEXT_SIZE_DIP = 14;

    private Integer sensorOrientation;
    private Classifier classifier;
    private Matrix inputToCropTranform;
    private Matrix cropToInputTransform;
    private BorderedText borderedText;

    private int crop_x;
    private int crop_y;
    private List<Classifier.Recognition> results;

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                TEXT_SIZE_DIP,
                getResources().getDisplayMetrics()
        );
        borderedText = new BorderedText(textSizePx);
        borderedText.setInteriorColor(Color.rgb(200, 200, 200));

        classifier = SVHNClassifier.create(getAssets());

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        inputBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

        inputToCropTranform = ImageUtils.getTransformationMatrix(
                CROP_SIZE.getWidth(), CROP_SIZE.getHeight(),
                INPUT_SIZE, INPUT_SIZE,
                sensorOrientation,
                false
        );

        cropToInputTransform = new Matrix();
        inputToCropTranform.invert(cropToInputTransform);

        crop_x = Math.max(0, previewWidth / 2 - CROP_SIZE.getWidth() / 2);
        crop_y = Math.max(0, previewHeight / 2 - CROP_SIZE.getHeight() / 2);

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderBox(canvas);
                        renderResults(canvas);
                        renderDebug(canvas);
                    }
                }
        );
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        Bitmap croppedBitmap = Bitmap.createBitmap(
                rgbFrameBitmap,
                crop_x, crop_y,
                CROP_SIZE.getWidth(), CROP_SIZE.getHeight(),
                new Matrix(), true
        );

        final Canvas canvas = new Canvas(inputBitmap);
        canvas.drawBitmap(croppedBitmap, inputToCropTranform, null);

        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(inputBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        results = classifier.recognizeImage(inputBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        cropCopyBitmap = Bitmap.createBitmap(inputBitmap);
                        requestRender();
                        readyForNextImage();
                    }
                });
    }

    @Override
    public void onSetDebug(boolean debug) {
        classifier.enableStatLogging(debug);
    }

    private void renderResults(final Canvas canvas) {
        Paint paint = new Paint();
        paint.setTextSize(50);
        paint.setColor(Color.argb(150, 0, 0, 0));
        canvas.drawRect(0, 0, canvas.getWidth(), 150, paint);

        StringBuilder msg = new StringBuilder();
        msg.append("Looks like ");
        if (results != null) {
            for (Classifier.Recognition result : results) {
                int d = result.getConfidence().intValue();
                if (d != 10) {
                    msg.append(Integer.toString(d));
                }
            }
        }

        paint = new Paint();
        paint.setTextSize(80);
        paint.setColor(Color.rgb(230, 230, 230));
        canvas.drawText(msg.toString(), 20, 100, paint);
    }

    private void renderBox(final Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.argb(100, 255, 0, 0));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(8);

        final boolean rotated = sensorOrientation % 180 == 90;

        float multiplier = Math.min(
                canvas.getHeight() / (float) (rotated ? previewWidth : previewHeight),
                canvas.getWidth() / (float) (rotated ? previewHeight : previewWidth)
        );

        Matrix frameToCanvasMatrix = ImageUtils.getTransformationMatrix(
                previewWidth,
                previewHeight,
                (int) (multiplier * (rotated ? previewHeight : previewWidth)),
                (int) (multiplier * (rotated ? previewWidth : previewHeight)),
                sensorOrientation,
                false
        );

        final RectF box = new RectF(
                crop_x,
                crop_y,
                crop_x + CROP_SIZE.getWidth(),
                crop_y + CROP_SIZE.getHeight()
        );

        frameToCanvasMatrix.mapRect(box);
        canvas.drawRect(box, paint);
    }

    private void renderDebug(final Canvas canvas) {
        final Bitmap copy = cropCopyBitmap;
        if (copy != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                    canvas.getHeight() - copy.getHeight() * scaleFactor
            );
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            lines.add("Crop size: " + CROP_SIZE.getWidth() + "x" + CROP_SIZE.getHeight());
            lines.add("Input size: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + " ms.");
            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
        }
    }
}
