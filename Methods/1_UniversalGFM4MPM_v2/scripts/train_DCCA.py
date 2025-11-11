def fit_unified_head_OVERLAP_from_uv(
    gA: nn.Module,
    data_loader_uv: DataLoader,
    d_u: int,
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 10,
    lambda_ent: float = 0.1,
    lambda_cons: float = 0.5,
    view_dropout: float = 0.3,
    noise_sigma: float = 0.0
) -> nn.Module:
    
    # ✅ Check what type of outputs gA produces
    gA_outputs_probs = isinstance(gA, MLPDropout)  # MLPDropout outputs probs
    
    # Build unified head
    gAB = PNHeadUnified(d=d_u).to(device)
    gAB_outputs_probs = True  # PNHeadUnified outputs probabilities
    
    opt = torch.optim.AdamW(gAB.parameters(), lr=lr, weight_decay=1e-4)
    eps = 1e-6
    
    for epoch in range(1, steps + 1):
        epoch_loss = 0.0
        epoch_loss_kd = 0.0
        epoch_loss_ent = 0.0
        epoch_loss_cons = 0.0
        batch_count = 0
        
        pbar = tqdm(data_loader_uv, desc=f"    [cls - 2] Epoch {epoch:3d}/{steps}", leave=False)
        
        for u, v, bmiss in pbar:
            u, v, bmiss = u.to(device), v.to(device), bmiss.to(device)
            
            with torch.no_grad():
                # Optional noise for robustness
                if noise_sigma > 0:
                    u_noisy = u + noise_sigma * torch.randn_like(u)
                    v_noisy = v + noise_sigma * torch.randn_like(v)
                else:
                    u_noisy, v_noisy = u, v
                
                # ✅ Get teacher predictions in PROBABILITY space
                teacher_output = gA(u_noisy)
                if gA_outputs_probs:
                    teacher = teacher_output.detach()
                else:
                    teacher = torch.sigmoid(teacher_output).detach()  # Convert logits to probs
            
            # View dropout: teach robustness when B is absent
            if torch.rand(1).item() < view_dropout:
                v_in = torch.zeros_like(v_noisy)
                b_in = torch.ones(u_noisy.size(0), 1, device=device)
            else:
                v_in = v_noisy
                b_in = bmiss
            
            # ✅ Student predictions (ensure probabilities)
            student_output = gAB(u_noisy, v_in, b_in)
            if gAB_outputs_probs:
                p_student = student_output
            else:
                p_student = torch.sigmoid(student_output)
            
            # ✅ Consistency target (ensure probabilities)
            cons_output = gAB(u_noisy, v, torch.zeros_like(b_in))
            if gAB_outputs_probs:
                p_cons = cons_output
            else:
                p_cons = torch.sigmoid(cons_output)
            
            # ✅ All losses now operate on probabilities [0,1]
            loss_kd = F.mse_loss(p_student, teacher)
            loss_ent = -(p_student * torch.log(p_student + eps) + 
                        (1 - p_student) * torch.log(1 - p_student + eps)).mean()
            loss_cons = F.mse_loss(p_student, p_cons.detach())
            
            loss = loss_kd + lambda_ent * loss_ent + lambda_cons * loss_cons
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            epoch_loss_kd += loss_kd.item()
            epoch_loss_ent += loss_ent.item()
            epoch_loss_cons += loss_cons.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'kd': f'{loss_kd.item():.4e}',
                'ent': f'{loss_ent.item():.4e}',
                'cons': f'{loss_cons.item():.4e}'
            })
        
        pbar.close()
        
        # Print epoch summary
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        avg_kd = epoch_loss_kd / batch_count if batch_count > 0 else 0.0
        avg_ent = epoch_loss_ent / batch_count if batch_count > 0 else 0.0
        avg_cons = epoch_loss_cons / batch_count if batch_count > 0 else 0.0
        
        print(f"    [cls - 2] Epoch {epoch:3d} | "
              f"loss={avg_loss:.4e} | "
              f"kd={avg_kd:.4e} | "
              f"ent={avg_ent:.4e} | "
              f"cons={avg_cons:.4e}")
    
    print(f"\n    [cls - 2] Training completed after {steps} epochs")
    return gAB.eval()