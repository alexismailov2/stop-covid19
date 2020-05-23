#include "VisualizeDangerIntersection.hpp"
#include "TimeMeasuring.hpp"

#include <opencv2/opencv.hpp>

namespace {
    auto getListOfDangerPairs(YOLOv3::Item::List const &list, float dangerDistance) -> std::vector <std::pair<int32_t, int32_t>>
    {
        std::vector <std::pair<int32_t, int32_t>> dangerPairs;
        auto getDistance = [&](auto i, auto j) {
            return std::sqrt(std::pow(list[i].boundingBox.x - list[j].boundingBox.x, 2) +
                             std::pow(list[i].boundingBox.y - list[j].boundingBox.y, 2));
        };
        for (auto i = 0; i < list.size(); ++i)
        {
            for (auto j = i + 1; j < list.size(); ++j)
            {
                if (getDistance(i, j) <= dangerDistance)
                {
                    dangerPairs.emplace_back(i, j);
                }
            }
        }
        return dangerPairs;
    }
} /// end namespace anonymous

void VisualizeDangerIntersection::draw(cv::Mat& frame,
                                       YOLOv3::Item::List const& list,
                                       float dangerDistance,
                                       cv::Scalar normalColor,
                                       cv::Scalar dangerColor,
                                       int32_t normalThickness,
                                       int32_t dangerThickness)
{
    TAKEN_TIME();
    auto listOfDangerPairs = getListOfDangerPairs(list, dangerDistance);
    for (int i = 0; i < list.size(); ++i)
    {
        bool isDangerDistance = std::find_if(listOfDangerPairs.cbegin(), listOfDangerPairs.cend(),
                                             [&](auto const& a){ return a.first == i || a.second == i; }) != listOfDangerPairs.cend();
        if (isDangerDistance)
        {
            cv::rectangle(frame, list[i].boundingBox, dangerColor, dangerThickness);
        }
        else
        {
            cv::ellipse(frame,cv::RotatedRect{list[i].boundingBox.tl(),
                                              cv::Point2f(list[i].boundingBox.br().x, list[i].boundingBox.tl().y),
                                              list[i].boundingBox.br()},
                                              normalColor,
                                              normalThickness);
        }
    }
    for (auto item : listOfDangerPairs)
    {
        auto const first = cv::Point(list[item.first].boundingBox.tl().x + list[item.first].boundingBox.width / 2,
                                     list[item.first].boundingBox.tl().y + list[item.first].boundingBox.height / 2);
        auto const second = cv::Point(list[item.second].boundingBox.tl().x + list[item.second].boundingBox.width / 2,
                                      list[item.second].boundingBox.tl().y + list[item.second].boundingBox.height / 2);
        cv::circle(frame, first, dangerThickness * 2, dangerColor, cv::FILLED);
        cv::circle(frame, second, dangerThickness * 2, dangerColor, cv::FILLED);
        cv::line(frame, first, second, dangerColor, dangerThickness);
    }
}