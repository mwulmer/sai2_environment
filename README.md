# sai2-environment
current usage:
/project_name
├── sai2_environment
│   ├── client.py
│   └── robot_env.py
└── train.py

def main():

    env = RobotEnv(
        name = 'pick_and_place',
        action_space = 'ee_position',
        render = True
    )

    env.reset()    
    
    env.step(np.array([-0.3, 0.3, 0.7]))        
                    
                    
if __name__ == "__main__":
    main()
