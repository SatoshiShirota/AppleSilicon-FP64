#ifndef FP64_TIMING_H
#define FP64_TIMING_H

#include <time.h>

/**
 * @~japanese
 * @brief 単調増加クロック（CLOCK_MONOTONIC）から現在時刻を取得する。
 * @return クロックの起点からの経過秒数。
 * @~english
 * @brief Read the current time from the monotonic clock (CLOCK_MONOTONIC).
 * @return Elapsed seconds since the clock origin.
 * @~
 */
static inline double fp64_monotonic_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

#endif
