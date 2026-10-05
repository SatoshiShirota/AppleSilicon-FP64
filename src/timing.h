#ifndef FP64_TIMING_H
#define FP64_TIMING_H

#include <time.h>

/**
 * @~japanese
 * @brief 単調な時計の時刻を返す。
 * @return 起点からの経過秒数。
 * @~english
 * @brief Return the time from a monotonic clock.
 * @return Elapsed seconds from the clock origin.
 * @~
 */
static inline double fp64_monotonic_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

#endif
