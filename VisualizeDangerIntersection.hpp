#pragma once

#include "YOLOv3.hpp"

class VisualizeDangerIntersection
{
public:
    static void draw(cv::Mat& frame,
                     YOLOv3::Item::List const& list,
                     float dangerDistance = 150,
                     cv::Scalar normalColor = {0x00, 0xFF, 0xFF},
                     cv::Scalar dangerColor = {0x00, 0x00, 0xFF},
                     int32_t normalThickness = 2,
                     int32_t dangerThickness = 3);
};
