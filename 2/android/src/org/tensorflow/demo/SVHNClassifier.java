/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.util.*;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class SVHNClassifier implements Classifier {
    private static final String TAG = "SVHNClassifier";

    private static final String MODEL_FILE = "file:///android_asset/svhn_classifier.pb";
    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "out_idx/Softmax";
    private static final int NUM_OUTPUTS = 6;
    private static final int NUM_CLASSES = 11;
    private static final int INPUT_SIZE = 96;
    private static final int IMAGE_MEAN = 100;
    private static final float IMAGE_STD = 1;

    private String[] outputNames;
    private boolean logStats = false;
    private TensorFlowInferenceInterface inferenceInterface;

    private SVHNClassifier() {
    }

    public static Classifier create(AssetManager assetManager) {
        SVHNClassifier c = new SVHNClassifier();
        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        c.outputNames = new String[NUM_OUTPUTS];
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            c.outputNames[i] = OUTPUT_NAME.replace("idx", Integer.toString(i));
        }

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        float[] floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = (val >> 16) & 0xFF;
            floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
            floatValues[i * 3 + 2] = (val & 0xFF);
        }

        inferenceInterface.feed(INPUT_NAME, floatValues, 1, INPUT_SIZE, INPUT_SIZE, 3);
        inferenceInterface.run(outputNames, logStats);

        ArrayList<Recognition> recognitions = new ArrayList<>();
        for (String outputName: this.outputNames) {
            float[] outputs = new float[NUM_CLASSES];
            inferenceInterface.fetch(outputName, outputs);

            int idx = 0;
            float max = 0;
            for (int i = 0; i < outputs.length; i++) {
                if (outputs[i] > max) {
                    idx = i;
                    max = outputs[i];
                }
            }

            recognitions.add(new Recognition(outputName, outputName, (float)idx, null));
        }
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
