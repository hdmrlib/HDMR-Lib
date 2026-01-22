import numpy as np

def mean_squared_error(true, pred):
    """
    Compute mean squared error between true and predicted tensors.
    
    Args:
        true: True tensor (numpy array or backend tensor)
        pred: Predicted tensor (numpy array or backend tensor)
    
    Returns:
        float: Mean squared error
    """
    # Convert to numpy if needed
    try:
        import torch
        if torch.is_tensor(true):
            true = true.detach().cpu().numpy()
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
    except (ImportError, AttributeError):
        pass
    
    try:
        import tensorflow as tf
        if isinstance(true, tf.Tensor):
            true = true.numpy()
        if isinstance(pred, tf.Tensor):
            pred = pred.numpy()
    except (ImportError, AttributeError):
        pass
    
    # Ensure numpy arrays
    true = np.asarray(true)
    pred = np.asarray(pred)
    
    # Compute MSE
    return float(np.mean((true - pred) ** 2))

def sensitivity_analysis(tensor, components, component_list, g0=None, return_dict=False):
    """
    Calculate individual and combined effects of specific components.
    
    Args:
        tensor: Original tensor (numpy array or backend tensor)
        components: Dictionary of component tensors from HDMR/EMPR decomposition
        component_list: List of component names to analyze (e.g., ['g1', 'g23', 'g_0'])
        g0: The zeroth-order component (scalar constant term), required if 'g_0' is in component_list
        return_dict: If True, returns dictionary; if False, prints and returns None
    
    Returns:
        dict or None: Dictionary containing:
            - 'individual_effects': Effect of each specified component
            - 'combined_effect': Total effect of all specified components together
    """
    # Helper function to convert component to numpy
    def _to_numpy(component_tensor):
        try:
            import torch
            if torch.is_tensor(component_tensor):
                return component_tensor.detach().cpu().numpy()
        except (ImportError, AttributeError):
            pass
        
        try:
            import tensorflow as tf
            if isinstance(component_tensor, tf.Tensor):
                return component_tensor.numpy()
        except (ImportError, AttributeError):
            pass
        
        return np.asarray(component_tensor)
    
    # Convert tensor to numpy
    tensor = _to_numpy(tensor)
    
    # Compute the squared norm of the original tensor
    original_norm_square = np.linalg.norm(tensor) ** 2
    
    # Calculate individual effects
    individual_effects = {}
    combined_norm_square = 0.0
    
    for comp_name in component_list:
        # Special handling for g_0 (constant term)
        if comp_name == 'g_0' or comp_name == 'g0':
            if g0 is None:
                print(f"Warning: g0 parameter required to analyze 'g_0' component")
                continue
            g0_square = float(g0) ** 2
            ratio = (g0_square / original_norm_square) * 100
            individual_effects['g_0'] = ratio
            combined_norm_square += g0_square
            continue
        
        # Regular component handling
        if comp_name not in components:
            print(f"Warning: Component '{comp_name}' not found in components dictionary")
            continue
        
        component_tensor = _to_numpy(components[comp_name])
        component_norm_square = np.linalg.norm(component_tensor) ** 2
        ratio = (component_norm_square / original_norm_square) * 100
        individual_effects[comp_name] = ratio
        combined_norm_square += component_norm_square
    
    # Calculate combined effect
    combined_effect = (combined_norm_square / original_norm_square) * 100
    
    if return_dict:
        return {
            'individual_effects': individual_effects,
            'combined_effect': combined_effect
        }
    else:
        # Print results
        print("\n" + "=" * 60)
        print(" " * 15 + "SENSITIVITY ANALYSIS")
        print("=" * 60)
        
        print(f"\nComponents analyzed: {', '.join(component_list)}")
        print("\nINDIVIDUAL EFFECTS:")
        print("-" * 60)
        for comp_name, ratio_value in individual_effects.items():
            print(f"   {comp_name:>10}: {ratio_value:>8.4f}%")
        
        print("\n" + "-" * 60)
        print(f"   {'COMBINED':>10}: {combined_effect:>8.4f}%")
        print("=" * 60 + "\n")
        
        return None