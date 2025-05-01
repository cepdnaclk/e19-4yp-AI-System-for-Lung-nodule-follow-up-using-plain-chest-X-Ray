import sys
from ct_xray_core import CTtoXrayConverter
from ct_xray_ui import CTtoXrayUI
from ct_xray_cli import CLI

def main():
    """Main entry point for the application"""
    # Check if running in CLI mode
    if len(sys.argv) > 1:
        # CLI mode
        cli = CLI()
        return cli.run(sys.argv[1:])
    else:
        # GUI mode
        try:
            import tkinter as tk
            root = tk.Tk()
            converter = CTtoXrayConverter()
            app = CTtoXrayUI(root,converter)
            root.mainloop()
            return 0
        except ImportError:
            print("Error: Tkinter not available. Run in CLI mode or install tkinter.")
            return 1

if __name__ == "__main__":
    sys.exit(main())