//libraries to use
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading;
class Program {
	
	//gets screen locations and dimensions
    [DllImport("User32.dll")]
    static extern IntPtr GetDC(IntPtr hwnd);
    [DllImport("User32.dll")]
    static extern int ReleaseDC(IntPtr hwnd, IntPtr dc);

    static void Main(string[] args) {
		//sets screen positions for desktop
        IntPtr desktop = GetDC(IntPtr.Zero);
		//sets font
		Font myFont = new Font("Arial", 40);

		// blank value for string
		String Line = "";
		try
		{	
			// reading the results file
			StreamReader sr = new StreamReader("C:\\Users\\omair\\Desktop\\neural networks\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\result.txt");
			Line = sr.ReadLine();
			sr.Close();
		}
		catch(Exception e)
		{
			Console.WriteLine(e.Message);
		}
		// displays the result text using a graphics object in a white rectangle
		using (Graphics g = Graphics.FromHdc(desktop)) {
            g.FillRectangle(Brushes.White, 0, 0, 175, 75);

			g.DrawString(Line, myFont, Brushes.Black, 0, 0);
		}
		
        ReleaseDC(IntPtr.Zero, desktop);
    }
}

