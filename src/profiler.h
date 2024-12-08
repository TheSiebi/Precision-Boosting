#ifndef PROFILER_H
#define PROFILER_H

#include <stdint.h>

#ifdef NPROFILER

#define PROFILE_SEGMENT_START(segment_name)
#define PROFILE_FUNCTION_START()
#define PROFILE_SEGMENTS_START(segment_name0, segment_name1)
#define PROFILE_FUNCTION_SEGMENT_START(segment_name)
#define PROFILE_SEGMENTS_SWITCH(segment_name)
#define PROFILE_SEGMENT_END()
#define PROFILE_SEGMENT_FUNCTION_END()
#define PROFILE_FUNCTION_END() 

#else //NPROFILER
#include <stdio.h>
#include "lib/tsc_x86.h"

struct profile_segment
{
    const char *name;
    uint64_t totalTime;
    uint64_t childTime;
    struct profile_segment *next;
    struct profile_segment *parent;
};

extern struct profile_segment *profile_segment_list;
extern struct profile_segment profile_segment_sentinel;
extern struct profile_segment *profile_segment_current;

#define CONCAT_IMP(start, end) start##end
#define CONCAT(start, end) CONCAT_IMP(start, end)
#define UNIQUE_VAR() CONCAT(prof, __COUNTER__)

#define PROFILE_SEGMENT_START_(segment_name, var_name) \
        static struct profile_segment var_name = {.name = segment_name}; \
        profile_segment_start(&var_name)

#define PROFILE_SEGMENT_START(segment_name) PROFILE_SEGMENT_START_(segment_name, UNIQUE_VAR())

#define PROFILE_FUNCTION_START() PROFILE_SEGMENT_START(__func__)

#define PROFILE_SEGMENTS_START_(segment_name0, segment_name1, var_name0, var_name1) \
        static struct profile_segment var_name0 = {.name = segment_name0}; \
        static struct profile_segment var_name1 = {.name = segment_name1}; \
        profile_segments_start(&var_name0, &var_name1)

#define PROFILE_SEGMENTS_START(segment_name0, segment_name1) \
        PROFILE_SEGMENTS_START_(segment_name0, segment_name1, UNIQUE_VAR(), UNIQUE_VAR())

#define PROFILE_FUNCTION_SEGMENT_START(segment_name) PROFILE_SEGMENTS_START(__func__, segment_name)

#define PROFILE_SEGMENT_SWITCH_(segment_name, var_name) \
        static struct profile_segment var_name = {.name = segment_name}; \
        profile_segments_switch(&var_name)

#define PROFILE_SEGMENTS_SWITCH(segment_name) PROFILE_SEGMENT_SWITCH_(segment_name, UNIQUE_VAR())

#define PROFILE_SEGMENT_END() profile_segment_end()

#define PROFILE_SEGMENT_FUNCTION_END() profile_segments_end()

#define PROFILE_FUNCTION_END() PROFILE_SEGMENT_END()


static inline void profile_segment_start(struct profile_segment *segment)
{
    if(segment->totalTime == 0)
    {
        segment->next = profile_segment_list;
        profile_segment_list = segment;
    }
    uint64_t counter = getRDTSCP();
    segment->totalTime -= counter;
    profile_segment_current->childTime -= counter;
    segment->parent = profile_segment_current;
    profile_segment_current = segment;
}

static inline void profile_segments_start(struct profile_segment *segment0, 
                                   struct profile_segment *segment1)
{
    if(segment0->totalTime == 0)
    {
        segment0->next = profile_segment_list;
        profile_segment_list = segment0;
    }
    if(segment1->totalTime == 0)
    {
        segment1->next = profile_segment_list;
        profile_segment_list = segment1;
    }
    uint64_t counter = getRDTSCP();
    segment0->totalTime -= counter;
    segment1->totalTime -= counter;
    profile_segment_current->childTime -= counter;
    segment0->childTime -= counter;
    segment0->parent = profile_segment_current;
    segment1->parent = segment0;
    profile_segment_current = segment1;
}

static inline void profile_segments_switch(struct profile_segment *newSegment)
{
    struct profile_segment *oldSegment = profile_segment_current;
    if(newSegment->totalTime == 0)
    {
        newSegment->next = profile_segment_list;
        profile_segment_list = newSegment;
    }
    uint64_t counter = getRDTSCP();
    oldSegment->totalTime += counter;
    newSegment->totalTime -= counter;
    newSegment->parent = oldSegment->parent;
    profile_segment_current = newSegment;
}

static inline void profile_segment_end()
{
    struct profile_segment *segment = profile_segment_current;
    profile_segment_current = segment->parent;
    uint64_t counter = getRDTSCP();
    segment->totalTime += counter;
    profile_segment_current->childTime += counter;
}

static inline void profile_segments_end()
{
    struct profile_segment *segment0 = profile_segment_current;
    struct profile_segment *segment1 = segment0->parent;
    uint64_t counter = getRDTSCP();
    profile_segment_current = segment1->parent;
    segment0->totalTime += counter;
    segment1->totalTime += counter;
    segment1->childTime += counter;
    profile_segment_current->childTime += counter;
}
#endif //NPROFILER
       
#ifdef __cplusplus
extern "C" {
#endif

void profiler_reset();
void profiler_segments_print(long flops16, long flops32, long flops64);

#ifdef __cplusplus
}
#endif

#endif //PROFILER_H
