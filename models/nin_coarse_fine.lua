-- Network-in-Network
require 'nn'
local utils = paths.dofile'utils.lua'

require 'cutorch'
cutorch.setDevice(opt.GPU)

function createModel()
    local b1 = nn.Sequential()
    local function Block1(...)
      local arg = {...}
      b1:add(nn.SpatialConvolution(...))
  -- model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
      b1:add(nn.ReLU(true))
      return b1
    end

    local b2 = nn.Sequential()
    local function Block2(...)
      local arg = {...}
      b2:add(nn.SpatialConvolution(...))
  -- model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
      b2:add(nn.ReLU(true))
      return b2
    end

    local b3 = nn.Sequential()
    local function Block3(...)
      local arg = {...}
      b3:add(nn.SpatialConvolution(...))
  -- model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
      b3:add(nn.ReLU(true))
      return b3
    end


    Block1(3,192,5,5,1,1,2,2)
    Block1(192,160,1,1)
    Block1(160,96,1,1)

    b2:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    b2:add(nn.Dropout(0.5))
    Block2(96,192,5,5,1,1,2,2)
    b21 = nn.Sequential()
    b21:add(nn.SpatialConvolution(192,64,1,1,1,1,0,0))
    b21:add(nn.ReLU(true))

    b22 = nn.Sequential()
    b22:add(nn.SpatialConvolution(192,96,1,1,1,1,0,0))
    b22:add(nn.ReLU(true))
    b22:add(nn.SpatialConvolution(96,128,3,3,1,1,1,1))
    b22:add(nn.ReLU(true))

    b23 = nn.Sequential()
    b23:add(nn.SpatialConvolution(192,16,1,1,1,1,0,0))
    b23:add(nn.ReLU(true))
    b23:add(nn.SpatialConvolution(16,32,5,5,1,1,2,2))
    b23:add(nn.ReLU(true))
    b24 = nn.Sequential()

    b24 = nn.Sequential()
    b24:add(nn.SpatialMaxPooling(3,3,1,1,1,1))
    b24:add(nn.SpatialConvolution(192,32,1,1,1,1,0,0))
    b24:add(nn.ReLU(true))
    mlp2 = nn.Concat(2)
    mlp2:add(b21):add(b22):add(b23):add(b24)
    b2:add(mlp2)
    Block2(256,192,1,1)
    Block2(192,192,1,1)
    b3:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
    b3:add(nn.Dropout(0.5))
-- b2:add(nn.SpatialFullConvolution(192, 192, 3, 3, 2, 2, 1,1,1,1))

    Block3(192,192,3,3,1,1,1,1)
    b31 = nn.Sequential()
    b31:add(nn.SpatialConvolution(192,64,1,1,1,1,0,0))
    b31:add(nn.ReLU(true))

    b32 = nn.Sequential()
    b32:add(nn.SpatialConvolution(192,96,1,1,1,1,0,0))
    b32:add(nn.ReLU(true))
    b32:add(nn.SpatialConvolution(96,128,3,3,1,1,1,1))
    b32:add(nn.ReLU(true))

    b33 = nn.Sequential()
    b33:add(nn.SpatialConvolution(192,16,1,1,1,1,0,0))
    b33:add(nn.ReLU(true))
    b33:add(nn.SpatialConvolution(16,32,5,5,1,1,2,2))
    b33:add(nn.ReLU(true))
    b34 = nn.Sequential()

    b34 = nn.Sequential()
    b34:add(nn.SpatialMaxPooling(3,3,1,1,1,1))
    b34:add(nn.SpatialConvolution(192,32,1,1,1,1,0,0))
    b34:add(nn.ReLU(true))
    mlp3 = nn.Concat(2)
    mlp3:add(b31):add(b32):add(b33):add(b34)
    b3:add(mlp3)
    --Block3(192,192,3,3,1,1,1,1)
    Block3(256,192,1,1)
    Block3(192,100,1,1) 

    model = nn.Sequential()
    model:add(b1)
    model:add(b2)
    model:add(b3)
    fenzhi1 = b1
    fenzhi2 = nn.Sequential():add(b1):add(b2):add(nn.SpatialFullConvolution(192, 192, 3, 3, 2, 2, 1,1,1,1))
    fenzhi3 = nn.Sequential():add(b1):add(b2):add(b3):add(nn.SpatialFullConvolution(100, 192, 3, 3, 4, 4, 1,1,3,3))
--image = torch.randn(5, 3, 32, 32)
    mlp = nn.Concat(2)
    mlp:add(fenzhi1)
    mlp:add(fenzhi2)
    mlp:add(fenzhi3)
--print(fenzhi1:forward(image):size())
--print(fenzhi2:forward(image):size())
--print(fenzhi3:forward(image):size())
--print(mlp:forward(image):size())
    model_before = nn.Sequential()
    model_before:add(mlp)
    model_before:add(nn.SpatialConvolution(480,100,1,1,1,1,0,0))
--print(model_before:forward(image):size())
    --utils.MSRinit(model_before)

    -- branch 1 : fine branch
    local fine_branch = nn.Sequential()
    fine_branch:add(nn.SpatialAveragePooling(32,32))
    fine_branch:add(nn.Reshape(100))  -- 100x1x1 ---> 100 ---> :dim()==1
    fine_branch:add(nn.LogSoftMax())

    -- branch 2 : coarse branch, min-pooling should be done here
    -- part 1, split 100x8x8 into 100 seperately 1x8x8
    local coarse_branch = nn.Sequential()
    coarse_branch:add(nn.SplitTable(-3))

    local coarse_branch_reshape_100 = nn.ParallelTable()
    for i=1, 100 do
        coarse_branch_reshape_100:add(nn.Reshape(1, 32, 32))
    end
    coarse_branch:add(coarse_branch_reshape_100)

    -- group the subclasses belong to the same superclass
    local hierarchy_path = paths.concat(opt.data,'fine_coarse_label_hierarchy.t7')
    local fine_coarse_label_hierachy = torch.load(hierarchy_path)

    local coarse_branch_hierachy_table = {}
    for i=1, 20 do
        coarse_branch_hierachy_table[i] = nn.ConcatTable()
        for j=1, 5 do
            coarse_branch_hierachy_table[i]:add(nn.SelectTable(fine_coarse_label_hierachy[i][j]))
        end
    end

    local coarse_branch_hierachy = nn.ConcatTable()
    for i=1, 20 do
        coarse_branch_hierachy:add(coarse_branch_hierachy_table[i])
    end
    coarse_branch:add(coarse_branch_hierachy)

    -- join the 5 seperate 1x8x8 --> 5x8x8
    local coarse_branch_join = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_join:add(nn.JoinTable(1,3))
    end
    coarse_branch:add(coarse_branch_join)

    -- minPooling
    local coarse_branch_minpooling = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_minpooling:add(nn.Min(1,3))
    end
    coarse_branch:add(coarse_branch_minpooling)

    -- reshape and join --> 20x8x8 --> averagepooling --> 20x1x1 --> 20
    local coarse_branch_reshape_20 = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_reshape_20:add(nn.Reshape(1, 32, 32))
    end
    coarse_branch:add(coarse_branch_reshape_20):add(nn.JoinTable(1,3)):add(nn.SpatialAveragePooling(32,32))

    coarse_branch:add(nn.Reshape(20))
    coarse_branch:add(nn.LogSoftMax())
    
    -- Concat fine_branch and coarse_branch, note the two branches are in parallel
    local branch = nn.ConcatTable()
    branch:add(fine_branch):add(coarse_branch)
    model_before:add(branch)
    utils.MSRinit(model_before)
    model_before:cuda()
    return model_before
end
