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

std::vector<std::string> noc_config(const std::string &topology,
        const std::string &routing, const size_t virtual_channels,
        const size_t width, const size_t height, const size_t cores_per_tile)
{
    return sanafe::booksim_config_from_noc(topology, routing,
            virtual_channels, width, height, cores_per_tile, 4);
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

TEST(ScheduleBooksimTest, MeshSquareSingleCore)
{
    const auto config = noc_config("mesh", "dimension_order", 0, 8, 8, 1);
    EXPECT_TRUE(has_entry(config, "topology = mesh"));
    EXPECT_TRUE(has_entry(config, "routing_function = dim_order"));
    EXPECT_TRUE(has_entry(config, "k = 8"));
    EXPECT_TRUE(has_entry(config, "c = 1"));
}

TEST(ScheduleBooksimTest, MeshRectangularUsesCmesh)
{
    const auto config = noc_config("mesh", "dimension_order", 0, 8, 4, 1);
    EXPECT_TRUE(has_entry(config, "topology = cmesh"));
    EXPECT_TRUE(has_entry(config, "routing_function = dor_no_express"));
    EXPECT_TRUE(has_entry(config, "x = 8"));
    EXPECT_TRUE(has_entry(config, "y = 4"));
}

TEST(ScheduleBooksimTest, MeshFourCoresUsesCmesh)
{
    const auto config = noc_config("mesh", "dimension_order", 0, 8, 8, 4);
    EXPECT_TRUE(has_entry(config, "topology = cmesh"));
    EXPECT_TRUE(has_entry(config, "c = 4"));
}

TEST(ScheduleBooksimTest, TorusSquareSingleCore)
{
    const auto config = noc_config("torus", "dimension_order", 0, 8, 8, 1);
    EXPECT_TRUE(has_entry(config, "topology = torus"));
    EXPECT_TRUE(has_entry(config, "routing_function = dim_order"));
    // Dimension-order routing on a torus splits the channels between phases.
    EXPECT_TRUE(has_entry(config, "num_vcs = 2"));
}

TEST(ScheduleBooksimTest, TorusInvalidGridThrows)
{
    EXPECT_THROW(noc_config("torus", "dimension_order", 0, 8, 4, 1),
            std::runtime_error);
    EXPECT_THROW(noc_config("torus", "dimension_order", 0, 8, 8, 4),
            std::runtime_error);
}

TEST(ScheduleBooksimTest, RoutingFunctionMapped)
{
    const auto mesh = noc_config("mesh", "xy_yx", 0, 8, 8, 1);
    EXPECT_TRUE(has_entry(mesh, "routing_function = xy_yx"));
    // xy_yx sends a packet one way or the other, so it needs two channels.
    EXPECT_TRUE(has_entry(mesh, "num_vcs = 2"));

    const auto concentrated = noc_config("mesh", "xy_yx", 0, 8, 4, 4);
    EXPECT_TRUE(has_entry(concentrated, "routing_function = xy_yx_no_express"));
}

TEST(ScheduleBooksimTest, VirtualChannelsOverride)
{
    const auto config = noc_config("mesh", "dimension_order", 4, 8, 8, 1);
    EXPECT_TRUE(has_entry(config, "num_vcs = 4"));

    EXPECT_THROW(
            noc_config("mesh", "xy_yx", 1, 8, 8, 1), std::runtime_error);
}

TEST(ScheduleBooksimTest, UnsupportedNocThrows)
{
    EXPECT_THROW(noc_config("dragonfly", "dimension_order", 0, 8, 8, 1),
            std::runtime_error);
    EXPECT_THROW(noc_config("mesh", "dimension_order", 0, 8, 8, 2),
            std::runtime_error);
    EXPECT_THROW(noc_config("mesh", "nonsense", 0, 8, 8, 1),
            std::runtime_error);
    // Valiant routing exists on a mesh and a torus, but not on a cmesh.
    EXPECT_THROW(noc_config("mesh", "valiant", 0, 8, 4, 4),
            std::runtime_error);
}

TEST(ScheduleBooksimTest, NonCycleModelsSkipBooksim)
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

TEST(ScheduleBooksimTest, CycleModelTorusTopology)
{
    sanafe::Scheduler scheduler;
    scheduler.timing_model = sanafe::timing_model_cycle_accurate;
    scheduler.noc_topology = "torus";
    describe_noc(scheduler, 4, 4, 1);
    EXPECT_NO_THROW(sanafe::schedule_create_threads(scheduler, 1));
    EXPECT_EQ(scheduler.booksim_instances.size(), 1U);
    sanafe::schedule_stop_all_threads(scheduler);
}
