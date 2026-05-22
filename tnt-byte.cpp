#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

class StructuralConfidenceDehazer {
public:
    StructuralConfidenceDehazer(
        float edgeWeight = 1.2f,
        float textureWeight = 1.0f,
        float entropyWeight = 0.8f,
        float reconstructionStrength = 1.4f,
        int blurRadius = 31
    ) : edgeWeight(edgeWeight),
        textureWeight(textureWeight),
        entropyWeight(entropyWeight),
        reconstructionStrength(reconstructionStrength),
        blurRadius(blurRadius) {}

    struct Results {
        cv::Mat restored;
        cv::Mat confidence;
        cv::Mat atmosphere;
        cv::Mat edgeConfidence;
        cv::Mat textureConfidence;
        cv::Mat entropyConfidence;
    };

    Results process(const cv::Mat& input) {
        cv::Mat image;
        input.convertTo(image, CV_32FC3, 1.0 / 255.0);

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat edgeConf = computeEdgeConfidence(gray);
        cv::Mat textureConf = computeTextureConfidence(gray);
        cv::Mat entropyConf = computeEntropyConfidence(gray);

        cv::Mat confidence = (
            edgeWeight * edgeConf +
            textureWeight * textureConf +
            entropyWeight * entropyConf
        );

        normalize(confidence);

        cv::Mat atmosphere = estimateAtmosphericField(confidence);

        cv::Mat restored = restoreImage(image, atmosphere, confidence);

        return {
            restored,
            confidence,
            atmosphere,
            edgeConf,
            textureConf,
            entropyConf
        };
    }

private:
    float edgeWeight;
    float textureWeight;
    float entropyWeight;
    float reconstructionStrength;
    int blurRadius;


    void normalize(cv::Mat& img) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        img = (img - minVal) / (maxVal - minVal + 1e-6f);
    }


    cv::Mat computeEdgeConfidence(const cv::Mat& gray) {
        cv::Mat gx, gy;

        cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

        cv::Mat magnitude;
        cv::magnitude(gx, gy, magnitude);

        cv::GaussianBlur(magnitude, magnitude, cv::Size(5, 5), 0);

        normalize(magnitude);

        return magnitude;
    }


    cv::Mat computeTextureConfidence(const cv::Mat& gray) {
        cv::Mat mean, sqMean;

        cv::GaussianBlur(gray, mean, cv::Size(7, 7), 3);

        cv::Mat squared = gray.mul(gray);
        cv::GaussianBlur(squared, sqMean, cv::Size(7, 7), 3);

        cv::Mat variance = sqMean - mean.mul(mean);

        cv::threshold(variance, variance, 0.0, 0.0, cv::THRESH_TOZERO);

        normalize(variance);

        return variance;
    }

    cv::Mat computeEntropyConfidence(const cv::Mat& gray, int windowSize = 9) {
        int pad = windowSize / 2;

        cv::Mat padded;
        cv::copyMakeBorder(
            gray,
            padded,
            pad,
            pad,
            pad,
            pad,
            cv::BORDER_REFLECT
        );

        cv::Mat entropyMap = cv::Mat::zeros(gray.size(), CV_32F);

        for (int y = 0; y < gray.rows; y++) {
            for (int x = 0; x < gray.cols; x++) {

                cv::Rect roi(x, y, windowSize, windowSize);
                cv::Mat patch = padded(roi);

                std::vector<int> hist(32, 0);

                for (int py = 0; py < patch.rows; py++) {
                    for (int px = 0; px < patch.cols; px++) {
                        float value = patch.at<float>(py, px);
                        int bin = std::min(31, static_cast<int>(value * 31.0f));
                        hist[bin]++;
                    }
                }

                float entropy = 0.0f;
                float total = static_cast<float>(windowSize * windowSize);

                for (int count : hist) {
                    if (count > 0) {
                        float p = count / total;
                        entropy -= p * std::log2(p + 1e-6f);
                    }
                }

                entropyMap.at<float>(y, x) = entropy;
            }
        }

        normalize(entropyMap);

        return entropyMap;
    }


    cv::Mat estimateAtmosphericField(const cv::Mat& confidence) {
        cv::Mat inverted = 1.0f - confidence;

        cv::Mat atmosphere;

        cv::GaussianBlur(
            inverted,
            atmosphere,
            cv::Size(blurRadius, blurRadius),
            0
        );

        normalize(atmosphere);

        return atmosphere;
    }

    cv::Mat directionalSharpen(const cv::Mat& image) {
        cv::Mat blurred;

        cv::GaussianBlur(image, blurred, cv::Size(0, 0), 2.0);

        cv::Mat sharpened;

        cv::addWeighted(image, 1.8, blurred, -0.8, 0, sharpened);

        cv::threshold(sharpened, sharpened, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(sharpened, sharpened, 0.0, 0.0, cv::THRESH_TOZERO);

        return sharpened;
    }


    cv::Mat restoreImage(
        const cv::Mat& image,
        const cv::Mat& atmosphere,
        const cv::Mat& confidence
    ) {
        cv::Mat sharpened = directionalSharpen(image);

        cv::Mat confidence3D;
        cv::Mat atmosphere3D;

        cv::Mat channels[] = { confidence, confidence, confidence };
        cv::merge(channels, 3, confidence3D);

        cv::Mat atmosChannels[] = { atmosphere, atmosphere, atmosphere };
        cv::merge(atmosChannels, 3, atmosphere3D);

        cv::Mat restored =
            image.mul(1.0f - atmosphere3D) +
            sharpened.mul(confidence3D * reconstructionStrength);

        cv::threshold(restored, restored, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(restored, restored, 0.0, 0.0, cv::THRESH_TOZERO);

        return restored;
    }
};


cv::Mat toDisplay(const cv::Mat& img) {
    cv::Mat normalized;

    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);

    normalized = (img - minVal) / (maxVal - minVal + 1e-6);

    cv::Mat output;
    normalized.convertTo(output, CV_8U, 255.0);

    return output;
}


int main() {

    std::string path = "foggy.jpg";

    cv::Mat image = cv::imread(path);

    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    StructuralConfidenceDehazer dehazer;

    auto results = dehazer.process(image);

    cv::imwrite("restored.png", toDisplay(results.restored));
    cv::imwrite("confidence_map.png", toDisplay(results.confidence));
    cv::imwrite("atmosphere_map.png", toDisplay(results.atmosphere));
    cv::imwrite("edge_confidence.png", toDisplay(results.edgeConfidence));
    cv::imwrite("texture_confidence.png", toDisplay(results.textureConfidence));
    cv::imwrite("entropy_confidence.png", toDisplay(results.entropyConfidence));

    std::cout << "Done." << std::endl;

    return 0;
}
