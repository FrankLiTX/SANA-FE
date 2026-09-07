// How the scheduler turns the NoC description into a Booksim configuration.
#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "chip.hpp"
#include "schedule.hpp"

// any helper functions go here
namespace
{
bool has_entry(const std::vector<std::string> &config, const std::string &entry)
{
    return std::find(config.begin(), config.end(), entry) != config.end();
}

void describe_noc(sanafe::Scheduler &scheduler, const size_t width,
        const size_t height, const size_t cores_per_tile)
{
    scheduler.noc_width_in_tiles = width;
    scheduler.noc_height_in_tiles = height;
    scheduler.max_cores_per_tile = cores_per_tile;
    scheduler.buffer_size = 4;
}
}

TEST(BooksimConfigFromNoc, SquareSingleCoreMeshStaysPlain)
{
    const auto config = sanafe::booksim_config_from_noc("mesh", 8, 8, 1, 4);
    EXPECT_TRUE(has_entry(config, "topology = mesh"));
    EXPECT_TRUE(has_entry(config, "k = 8"));
    EXPECT_TRUE(has_entry(config, "c = 1"));
}

TEST(BooksimConfigFromNoc, RectangularMeshUsesConcentratedMesh)
{
    const auto config = sanafe::booksim_config_from_noc("mesh", 8, 4, 1, 4);
    EXPECT_TRUE(has_entry(config, "topology = cmesh"));
    EXPECT_TRUE(has_entry(config, "x = 8"));
    EXPECT_TRUE(has_entry(config, "y = 4"));
}

TEST(BooksimConfigFromNoc, FourCoresPerTileUsesConcentratedMesh)
{
    const auto config = sanafe::booksim_config_from_noc("mesh", 8, 8, 4, 4);
    EXPECT_TRUE(has_entry(config, "topology = cmesh"));
    EXPECT_TRUE(has_entry(config, "c = 4"));
}

TEST(BooksimConfigFromNoc, RejectsWhatBooksimCannotModel)
{
    EXPECT_THROW(sanafe::booksim_config_from_noc("torus", 8, 8, 1, 4),
            std::runtime_error);
    EXPECT_THROW(sanafe::booksim_config_from_noc("mesh", 8, 8, 2, 4),
            std::runtime_error);
}

TEST(ScheduleCreateThreads, OtherModelsNeverBuildBooksim)
{
    for (const auto model :
            {sanafe::timing_model_simple, sanafe::timing_model_detailed})
    {
        sanafe::Scheduler scheduler;
        scheduler.timing_model = model;
        // Two cores per tile is something Booksim cannot represent.
        describe_noc(scheduler, 8, 8, 2);
        EXPECT_NO_THROW(sanafe::schedule_create_threads(scheduler, 0));
        EXPECT_TRUE(scheduler.booksim_instances.empty());
    }
}

TEST(ScheduleBooksimTest, CycleModelInstancePerThread)
{
    sanafe::Scheduler scheduler;
    scheduler.timing_model = sanafe::timing_model_cycle_accurate;
    describe_noc(scheduler, 4, 4, 1);
    sanafe::schedule_create_threads(scheduler, 2);
    EXPECT_EQ(scheduler.booksim_instances.size(), 2U);

    sanafe::schedule_stop_all_threads(scheduler);
    EXPECT_TRUE(scheduler.booksim_instances.empty());
}

TEST(ScheduleBooksimTest, CycleModelUnsupportedNocThrows)
{
    sanafe::Scheduler scheduler;
    scheduler.timing_model = sanafe::timing_model_cycle_accurate;
    describe_noc(scheduler, 8, 8, 2);
    EXPECT_THROW(
            sanafe::schedule_create_threads(scheduler, 1), std::runtime_error);
}
