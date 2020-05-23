#include <opencv2/opencv.hpp>

#include <vector>

struct Item
{
    cv::Rect boundingBox;
    bool dangerDistanceToAnother{};
    using List = std::vector<Item>;
};

auto getListOfDangerPairs(Item::List const& list, float dangerDistance) -> std::vector<std::pair<int32_t, int32_t>>
{
    std::vector<std::pair<int32_t, int32_t>> dangerPairs;
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

void visualizeDangerPairs(cv::Mat& frame, Item::List const& list, float dangerDistance, cv::Scalar normalColor = {0x00, 0xFF, 0xFF}, cv::Scalar dangerColor = {0x00, 0x00, 0xFF}, int32_t normalThickness = 2, int32_t dangerThickness = 3)
{
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
            cv::ellipse(frame, cv::RotatedRect{list[i].boundingBox.tl(), cv::Point2f(list[i].boundingBox.br().x,
                        list[i].boundingBox.tl().y), list[i].boundingBox.br()}, normalColor, normalThickness);
        }
    }
    for (auto item : listOfDangerPairs)
    {
        auto const first = cv::Point{list[item.first].boundingBox.tl().x + list[item.first].boundingBox.width / 2,
                                     list[item.first].boundingBox.tl().y + list[item.first].boundingBox.height / 2};
        auto const second = cv::Point{list[item.second].boundingBox.tl().x + list[item.second].boundingBox.width / 2,
                                      list[item.second].boundingBox.tl().y + list[item.second].boundingBox.height / 2};
        cv::circle(frame, first, dangerThickness * 2, dangerColor, cv::FILLED);
        cv::circle(frame, second, dangerThickness * 2, dangerColor, cv::FILLED);
        cv::line(frame, first, second, dangerColor, dangerThickness);
    }
}

constexpr auto DANGER_DISTANCE = 50;

int main()
{
    cv::namedWindow("Test distances");
    while(1)
    {
        cv::Mat frame = cv::Mat::zeros(500, 500, CV_8UC3);
        auto points = Item::List(10);
        for (auto &item : points)
        {
            item.boundingBox = cv::Rect{std::rand() % 400, std::rand() % 400, 20, 30};
        }
        visualizeDangerPairs(frame, points, DANGER_DISTANCE);

        cv::imshow("Test distances", frame);
        cv::waitKey(1000);
    }
    return 0;
}
