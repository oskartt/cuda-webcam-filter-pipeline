// Create filter kernels used in the pipeline.
// Stage 1 applies a blur filter and Stage 2 applies a sharpen filter.
cv::Mat blurKernel = cuda_filter::FilterUtils::createFilterKernel(
    cuda_filter::FilterType::BLUR, 3, 1.0f);

cv::Mat sharpenKernel = cuda_filter::FilterUtils::createFilterKernel(
    cuda_filter::FilterType::SHARPEN, 3, 1.0f);

PLOG_INFO << "Filter pipeline enabled: blur -> sharpen";
PLOG_INFO << "Wipe transition enabled";
PLOG_INFO << "Press 'ESC' to exit";

cv::Mat frame;

// Intermediate buffers used to store results between pipeline stages.
cv::Mat stage1;
cv::Mat stage2;
cv::Mat transitionOutput;

double fps = 0.0;
int frameCount = 0;
double startTime = static_cast<double>(cv::getTickCount());

// Transition progress value.
// 0.0 = original image, 1.0 = fully processed image.
float transition = 0.0f;

while (true)
{
    if (!inputHandler.readFrame(frame))
    {
        PLOG_ERROR << "Failed to read frame";
        break;
    }

    // Start timing the complete filter pipeline.
    const double pipelineStart = static_cast<double>(cv::getTickCount());

    // Pipeline Stage 1:
    // Apply blur filter to the original frame.
    cuda_filter::applyFilterGPU(frame, stage1, blurKernel);

    // Pipeline Stage 2:
    // Apply sharpen filter to the blurred image.
    cuda_filter::applyFilterGPU(stage1, stage2, sharpenKernel);

    const double pipelineEnd = static_cast<double>(cv::getTickCount());

    // Calculate total execution time of the filter pipeline.
    const double pipelineTime =
        (pipelineEnd - pipelineStart) / cv::getTickFrequency();

    // Create output image used for wipe transition visualization.
    transitionOutput.create(frame.size(), frame.type());

    // Calculate current wipe position across the frame.
    int wipeX = static_cast<int>(frame.cols * transition);

    for (int y = 0; y < frame.rows; y++)
    {
        for (int x = 0; x < frame.cols; x++)
        {
            // Left side displays processed pipeline output.
            // Right side displays the original image.
            if (x < wipeX)
                transitionOutput.at<cv::Vec3b>(y, x) = stage2.at<cv::Vec3b>(y, x);
            else
                transitionOutput.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
        }
    }

    // Move wipe transition from left to right.
    transition += 0.01f;

    // Restart transition after reaching the end of the frame.
    if (transition > 1.0f)
        transition = 0.0f;

    // Calculate frames per second.
    frameCount++;

    double now = static_cast<double>(cv::getTickCount());

    if ((now - startTime) / cv::getTickFrequency() >= 1.0)
    {
        fps = frameCount;
        frameCount = 0;
        startTime = now;
    }

    // Display real-time pipeline performance information.
    std::string text =
        "Pipeline FPS: " + std::to_string(static_cast<int>(fps)) +
        " Time: " + std::to_string(pipelineTime * 1000).substr(0, 5) + "ms";

    cv::putText(transitionOutput, text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 0), 2);

    cv::putText(transitionOutput, "Pipeline: blur -> sharpen",
                cv::Point(10, transitionOutput.rows - 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 0), 2);

    cv::putText(transitionOutput, "Wipe transition",
                cv::Point(10, transitionOutput.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 0), 2);

    // Display final output frame.
    inputHandler.displayFrame(transitionOutput);

    if (cv::waitKey(1) == 27)
    {
        break;
    }
}
