/*!
 * @copyright 2018 Bonn-Rhein-Sieg University
 *
 * @author Minh Nguyen
 *
 */
#include <vector>
#include <opencv2/imgproc.hpp>
#include <mas_perception_libs/bounding_box_2d.h>

namespace mas_perception_libs
{

void
BoundingBox2D::updateBox(const cv::Rect &pNewRect)
{
    mX = pNewRect.x;
    mY = pNewRect.y;
    mWidth = pNewRect.width;
    mHeight = pNewRect.height;
}

void
drawLabeledBoxes(cv::Mat &pImage, std::vector<BoundingBox2D> pBoundingBoxes, int pThickness, double pFontScale)
{
    cv::Size imageSize = pImage.size();
    for (auto &pBoundingBox : pBoundingBoxes)
    {
        // fit box to within image boundaries
        cv::Rect boxRect = fitBoxToImage(imageSize, pBoundingBox.getCvRect());

        // fit label into within image boundaries
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(pBoundingBox.mLabel, cv::FONT_HERSHEY_PLAIN,
                pFontScale, pThickness, &baseline);
        if (boxRect.y < textSize.height) boxRect.y = textSize.height;

        // draw box
        cv::rectangle(pImage, boxRect, pBoundingBox.mColor, pThickness);

        // draw label with background
        cv::Point textTopLeft = cv::Point(boxRect.x, boxRect.y - textSize.height);
        cv::Point textBottomRight = cv::Point(boxRect.x + textSize.width, boxRect.y + baseline);
        cv::rectangle(pImage, textTopLeft, textBottomRight, pBoundingBox.mColor, cv::FILLED);
        cv::putText(pImage, pBoundingBox.mLabel, cv::Point(boxRect.x, boxRect.y),
                    cv::FONT_HERSHEY_PLAIN, pFontScale, CV_RGB(255, 255, 555), 1);
    }
}

void
fitBoxToImage(const cv::Size &pImageSize, BoundingBox2D& pBox, int pSizeOffset)
{
    cv::Rect newBox = fitBoxToImage(pImageSize, pBox.getCvRect(), pSizeOffset);
    pBox.updateBox(newBox);
}

cv::Rect
fitBoxToImage(const cv::Size &pImageSize, cv::Rect pBox, int pSizeOffset)
{
    if (pSizeOffset)
    {
        // if size offset is specified, expand bounding box by x offset pixels
        pBox.x -= pSizeOffset;
        pBox.y -= pSizeOffset;
        pBox.width += 2 * pSizeOffset;
        pBox.height += 2 * pSizeOffset;
    }

    // check if box is outside of image, should be efficient since vectorized
    cv::Rect image_rect(0, 0, pImageSize.width, pImageSize.height);
    if ((pBox & image_rect) == pBox)
        return pBox;

    if (pBox.x < 0)
        pBox.x = 0;

    if (pBox.y < 0)
        pBox.y = 0;

    if (pBox.x + pBox.width >= pImageSize.width)
        pBox.width = pImageSize.width - pBox.x - 1;

    if (pBox.y + pBox.height >= pImageSize.height)
        pBox.height = pImageSize.height - pBox.y - 1;

    return pBox;
}

cv::Mat
cropImage(cv::Mat &pImage, BoundingBox2D &pBox, int pOffset, bool pCopy)
{
    cv::Rect cvBox = pBox.getCvRect();
    return cropImage(pImage, cvBox, pOffset, pCopy);
}

cv::Mat
cropImage(cv::Mat &pImage, const std::vector<cv::Point2f> &pVertices, int pOffset, bool pCopy)
{
    cv::Rect roiRect = cv::boundingRect(cv::Mat(pVertices));
    return cropImage(pImage, roiRect, pOffset, pCopy);
}

cv::Mat
cropImage(cv::Mat &pImage, cv::Rect &pRoiRect, int pOffset, bool pCopy)
{
    auto roiRect = fitBoxToImage(pImage.size(), pRoiRect, pOffset);
    cv::Mat croppedImage(pImage, roiRect);
    if (pCopy)
    {
        cv::Mat croppedCopy;
        croppedImage.copyTo(croppedCopy);
        return croppedCopy;
    }
    return croppedImage;
}

}   // namespace mas_perception_libs
