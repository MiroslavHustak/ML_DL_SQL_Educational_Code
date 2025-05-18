namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

module Educational_Code_TorchSharp =  

    let learninTensorOperations() = 

        let t = torch.tensor(35) //torch.tensor(35) creates a 0-dimensional tensor (a scalar), hodnota skalaru 35
        let t = torch.tensor([| 35 |]) // 1D tensor, vysledek je tensor([35])
        let t = torch.ones(3, 4, 12)  //naplneni 1.0, neb aji kdyz to intellisense neukazuje, type defaultne Float32            
        let t = torch.zeros(3, 4, 12)  //naplneni 0.0
        let t = torch.zeros(3, 4, 12, dtype=torch.int32) //naplneni 0 integer // intellisense stale nic neukazuje
        let t = torch.empty(3, 4, 12)  //nahodne naplneni cimkoliv, co je zrovna v pameti (neni to random)
        //torch.full(size: seq<int>, value: Scalar, dtype: optional<torch.dtype> = None, device: optional<torch.device> = None) -> Tensor
        //Scalar is a wrapper type for individual numeric values (like int, float, etc.). ToScalar() converts float32 (23.17f) into a Scalar that torch.full can understand.
        let t1 = torch.full(4, 8, 4, 23.17f.ToScalar()) //naplneni matice 4, 8, 4 hodnotou float 23.17,

        torch.zeros(4, 4).print() |> ignore<torch.Tensor> //to same, jako viz nize - vytiskne cely tensor na obrazovku
        printf "%A" <| torch.zeros(4, 4)

        let t = torch.zeros(4,4, dtype=torch.int32)
        t.[0L,0L].print() |> ignore<torch.Tensor> 
        let item : int = t.[0L,0L].item<int>() //analogie Array.item 0
        t.[0L,0L] <- torch.tensor(35) //podobne jako u Array, mutable preplacnuti prvku

        // Normal (Gaussian) distribution
        torch.randn(3L,4L) |> ignore<torch.Tensor> 
        use gaussian = new Normal(torch.tensor(3L), torch.tensor(4L)) 
        (*
        torch.randn(3L, 4L): Generates a 3x4 tensor of random numbers from a standard normal distribution (mean = 0, std dev = 1).

        tensor([[ 0.4563, -0.2345,  1.1234, -0.5432],
        [-1.2345,  0.2345, -0.7890,  0.6789],
        [ 0.1111, -0.2222,  0.3333, -0.4444]])
        Values are both positive and negative, centred around mean = 0, standard deviation = 1
                      
        new Normal(torch.tensor(3L), torch.tensor(4L)): This creates a normal distribution with: Mean (µ) = 3 Standard Deviation (σ) = 4        
        tensor([[ 5.67,  0.32,  2.54,  8.12],
        [ 3.76,  1.23, -2.78,  7.91],
        [ 4.02,  2.94,  0.48,  5.13]])
        
        *)

        //Uniform Distribution
        torch.rand(3,4) |> ignore<torch.Tensor> //uniform distribution between 0 (inclusive) and +1 (exclusive).
        use uniform = new Uniform(torch.tensor(3), torch.tensor(4))

        (*
        torch.rand(3,4)  Generates a 3x4 tensor of random numbers from a uniform distribution. Values are strictly between 0 and 1 ([0, 1)).
        tensor([[0.1234, 0.5678, 0.9101, 0.3456],
        [0.7890, 0.2345, 0.6789, 0.4567],
        [0.1123, 0.3345, 0.5567, 0.7789]])        

        new Uniform(torch.tensor(3), torch.tensor(4)) This creates a uniform distribution with Lower bound = 3 Upper bound = 4
        tensor([[3.24, 3.81, 3.02, 3.90],
        [3.67, 3.15, 3.43, 3.74],
        [3.57, 3.18, 3.99, 3.65]])
        *)

        // Uniform distribution between [100,110]
        torch.rand(3,4) * torch.tensor(10) + torch.tensor(100) |> ignore<torch.Tensor> 

        let arr = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f; 7.0f; 8.0f; 9.0f; 10.0f |]
        let tArr = torch.from_array(arr)            
           
        arr.[4] <- 100.0f           
        tArr.[6] <- 200
        arr |> ignore<float32 array>   //pozor, pointery, zmena arr zmeni tArr a naopak, uz jsem si odvykl v F#

        let arr = [| 1.0f; 2.0f; 3.0f |]
        let tArr = torch.from_array(arr).clone()  // This makes it independent
            
        arr.[1] <- 100.0f
        // Tensor tArr remains unchanged

        //a naopak tensor to array takto slozite
        let a = t.data<single>().ToArray()

        //arange() creates a 1D tensor with numbers ranging from a min to a max, exclusive of the max. You can provide the step value, or let it be the default, which is 1  
        //arange znamena array range
        torch.arange(3L,14L) |> ignore<torch.Tensor>  //vyjimka u arrange - the default element type is not float32, it's inferred from the arguments. 
        //tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])
        torch.arange(3.0f, 5.0f, step=0.1) |> ignore<torch.Tensor>

        //There is no way to make arange produce anything but a 1D tensor, so a common thing to do is to reshape it as soon as it's created.
        torch.arange(3.0f, 5.0f, step=0.1f).reshape(4,5) |> ignore<torch.Tensor>
        (*
        tensor([[3.0000, 3.1000, 3.2000, 3.3000, 3.4000],
        [3.5000, 3.6000, 3.7000, 3.8000, 3.9000],
        [4.0000, 4.1000, 4.2000, 4.3000, 4.4000],
        [4.5000, 4.6000, 4.7000, 4.8000, 4.9000]])
        *)

        let position = torch.arange(5L, dtype = torch.float32).unsqueeze(1)
        //If seqLen is 5L, the output of torch.arange is tensor([0., 1., 2., 3., 4.])
        //.unsqueeze(1) adds a new dimension at index 1 a misto rozmeru [|5|] zrobi [|5,  1|]
       (*
        [[0.],
        [1.],
        [2.],
        [3.],
        [4.]]  // Shape: [5, 1]
        *)

        let a = torch.ones(3,4)
        let b = torch.zeros(3,4)
        let c = torch.tensor(5)
        a * c + b |> ignore<torch.Tensor>
        a.mul_(c).add_(b) |> ignore<torch.Tensor> //mul = multiplication
        //fungovani a performance obou vyrazu viz docs

        let a = torch.full(4L, 4L, (17).ToScalar())
        let b = torch.full(4L, 4L, (12).ToScalar())
        
        (a * b).print() |> ignore<torch.Tensor> //Each element in a is multiplied by the corresponding element in b //Shape compatibility: Both tensors must be the same shape or broadcastable.
        (*
        tensor([[204, 204, 204, 204],
        [204, 204, 204, 204],
        [204, 204, 204, 204],
        [204, 204, 204, 204]])
        *)

        (a.mm(b)) |> ignore<torch.Tensor> //RuntimeError: self.size(-1) must match mat2.size(-2) 
        //.mm() stands for matrix multiplication.
        //For matrix multiplication to be valid, the number of columns of the first matrix must match the number of rows of the second matrix.

        (*
        In the coin toss scenario, there were two categories -- yes/no, true/false, 0/1, etc.
        A more general class of distributions support N different categories - 'Categorical,'. 
        The length of the probabilities tensor tells the Categorical class how many categories there are. The categories are represented as integers in the range [0..N].
        
        let cat = Categorical(torch.tensor([|0.1f; 0.7f; 0.1f; 0.1f|]))
        cat.sample(4L)
        
        'Binomial' class for binary distributions, 'Multinomial' class for categorical distributions that aren't binary. 

        In a Binomial distribution, you only have two outcomes (like a coin flip — heads or tails).

        In a Multinomial distribution, you have more than two outcomes (like rolling a die with multiple sides or drawing from a bag of coloured balls).
         
        let mult = Multinomial(100, torch.tensor([|0.1f; 0.7f; 0.1f; 0.1f|]))       
        100 trials — This is the total number of "draws" or "events" you are sampling.

        A probability distribution represented as a tensor: [0.1, 0.7, 0.1, 0.1].

        The tensor [0.1, 0.7, 0.1, 0.1] represents: 10% chance for event 0, 70% chance for event 1, 10% chance for event 2, 10% chance for event 3
        
        *)

        // Linear applies an affine linear transformation to the incoming data: y = x*A^T + b 
        // x input data (usually a matrix or vector), A^T transpose of matrix A
        // affine = linear transformation (nasobeni x a matice) + translation (pridani vektoru b)
        
        let qkvProjection = Linear (100, 300) // 100 vstupnich elementu, 300 vystupnich elementu
        qkvProjection

        (*
        ### ✅ **Complete List of Steps for Training a Model**:
        
        1️⃣ **Prepare the Data**
        
        * Load input data (`dataBatch`) and labels (`resultBatch`).
        
        2️⃣ **Initialize the Model**
        
        * Define the model architecture and layers.
        
        3️⃣ **Define the Loss Function**
        
        * Choose a loss function (e.g., MSE, Cross Entropy).
        
        4️⃣ **Choose an Optimizer**
        
        * Select an optimizer (e.g., SGD, Adam) and link it to the model parameters.
        
        ---
        
        ### 🔄 **Training Loop (For Each Epoch):**
        
        5️⃣ **Forward Pass (Inference)**
        
        * Pass the data through the model to get predictions.
        
        6️⃣ **Compute the Loss**
        
        * Compare predictions with actual labels using the loss function.
        
        7️⃣ **Backward Pass (Backpropagation)**
        
        * Compute the gradients of the loss with respect to each parameter.
        
        8️⃣ **Update the Weights**
        
        * Use the optimizer to adjust the model parameters based on gradients.
        
        9️⃣ **Zero the Gradients**
        
        * Clear gradients to prevent accumulation.
        
        🔟 **Repeat the Loop**
        
        * Perform steps 5️⃣ to 9️⃣ for many epochs until the model converges.
        
        ---
        
        ### 🏁 **After Training:**
        
        1️⃣1️⃣ **Evaluate the Model**
        
        * Test the model with unseen data to measure performance.
        
        1️⃣2️⃣ **Save the Model (Optional)**
        
        * Store the trained model for future use.     
        
        *)

