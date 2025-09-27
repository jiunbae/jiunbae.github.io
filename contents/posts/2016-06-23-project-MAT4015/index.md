---
title: "MAT4015: Fundamentals of applied probability and random processes"
description: "MAT4015: Fundamentals of applied probability and random processes"
date: 2016-06-23
slug: /MAT4015
tags: [curriculum]
heroImage: 
heroImageAlt: 
---

# Object tracking

Object tracking based on color histogram and real-time back projection

## Process

1. Modeling color histogram of selected object
2. Calculate histogram similarity
3. Back projection
4. Mean shift process (clustering of pixels with back projected weights)
5. Update object position and parameters after tracking

---

**Additional functions**

- Dynamic navigation area
- Dynamic change of object color histogram
- Dynamic change of object size


## Codes

- **Calculate color histogram**

```C++
double * hists = (double*)calloc(param.hist_bins, sizeof(double)), hist_size = 256 / param.hist_bins;
int eWidth = min(rc.x + rc.width, img.size().width),
    eHeight = min(rc.y + rc.height, img.size().height);
for (int i = rc.x; i < eWidth; ++i)
    for (int j = rc.y; j < eHeight; ++j)
        hists[matrixAt(img, i, j) / (int)hist_size]++;

double total = 0;
for (int i = 0; i < param.hist_bins; ++i)
    total += hists[i] * hists[i];
total = sqrt(total);
for (int i = 0; i < param.hist_bins; ++i)
    hists[i] /= total;

return hists;
```

- **Back projection**

```
Mat imx = hsv.clone();
for (int i = 0; i < hsv.cols; ++i)
    for (int j = 0; j < hsv.rows; ++j)
    {
        w = this->objectHists[matrixAt(imx, i, j) / param.hist_bins];
        double pixel[] = { 255 * w, 255 * w, 255 * w };
        matrixSet(imx, i, j, pixel);
    }
```

- **Mean shift**
```
Mat hsv;
cvtColor(img, hsv, CV_BGR2HSV);
Rect nRect = myRect, temp, bRect = myRect;
double nX = 0, nY = 0, tW = 0, w;
do {
    tW = 0, nX = 0; nY = 0; myRect = nRect;

    // set searching area
    temp = Rect(max(nRect.x - (int)param.search_range, 0), max(nRect.y - (int)param.search_range, 0),
        min(nRect.width + (int)param.search_range * 2, hsv.rows), min(nRect.height + (int)param.search_range * 2, hsv.cols));
    double sRatioWidth = temp.width / param.sampling, sRatioHeight = temp.height / param.sampling;
    for (int i = 0; i <= temp.width / 2; i+=sRatioWidth)
        for (int j = 0; j <= temp.height / 2; j+=sRatioHeight)
        {
            w = mySimilarity(hsv, Rect(temp.x + (temp.width / 2) + i, temp.y + (temp.height / 2) + j, nRect.width, nRect.height), this->objectHists);
            tW += w;
            nX += w * (temp.x + (temp.width / 2) + i);
            nY += w * (temp.y + (temp.height / 2) + j);
            if (i != 0 || j != 0)
            {
                w = mySimilarity(hsv, Rect(temp.x + (temp.width / 2) - i, temp.y + (temp.height / 2) - j, nRect.width, nRect.height), this->objectHists);
                tW += w;
                nX += w * (temp.x + (temp.width / 2) - i);
                nY += w * (temp.y + (temp.height / 2) - j);
            }
        }
    nX /= tW;
    nY /= tW;

    // get mean of w, if w > 0.25 make range narrow, or not make extend;
    double twRatio = tW / ((temp.width / sRatioWidth) * (temp.height / sRatioHeight));
    param.search_range *= (twRatio > (EXTEND_LIMIT) ? 1 - twRatio : (1 - EXTEND_LIMIT) +twRatio);

    nRect = Rect(max((int)nX - nRect.width / 2, 0), max((int)nY - nRect.height / 2, 0), myRect.width, myRect.height);
} while (sqrt(pow(myRect.x - nRect.x, 2) + pow(myRect.y - nRect.y, 2)) > param.search_range);
```
