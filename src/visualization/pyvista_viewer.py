import pyvista as pv
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class Line:
    scalar: str
    p1:tuple[int]
    p2:tuple[int]
    axis:int
    label: str
    resolution:int
    data_type:str 
    pv_line: pv.charts.LinePlot2D
    _set_mesh_display_called = False
class Pyvista_Visualizer():
    plotter: pv.Plotter
    '''Pyvista Plotter class that controls all animation'''
    mesh:pv.StructuredGrid
    def __init__(self,D:int,origin:tuple[float,float,float],domain_size:tuple[float,float,float],grid_shape:tuple[int,int,int],subplot_shape,**pyvista_Plotter_kwargs):
        self.subplot_shape =subplot_shape
        self.charts:dict[tuple[int],pv.Chart2D] = {}
        self.lines:dict[tuple[int],dict[str,Line]] = {}
        self.origin = origin
        self.domain_size = domain_size
        
        assert  D == 2 or D == 3
        coord_vectors = [np.linspace(origin[i], origin[i] + domain_size[i], grid_shape[i]) if i < D else np.linspace(0, 0, grid_shape[i]) for i in range(3)]
        m = np.meshgrid(*coord_vectors,indexing = 'ij')
        xx,yy,zz = m[0],m[1],m[2]
        self.mesh = pv.StructuredGrid(xx, yy, zz)
        self.plotter = pv.Plotter(shape = subplot_shape,**pyvista_Plotter_kwargs)
        
        
    @property
    def point_data(self):
        return self.mesh.point_data
    
    @property
    def cell_data(self):
        return self.mesh.cell_data
    
    def set_mesh_display(self,to_plot:str,subplot:tuple[int,int] = (0,0),**kwargs):
        assert len(subplot) == 2
        self._set_mesh_display_called = True
        plotter = self.plotter
        plotter.subplot(*subplot)
        plotter.add_mesh(self.mesh, scalars = to_plot,**kwargs)
        plotter.view_xy()
    
    def show(self):
        if not self._set_mesh_display_called:
            to_plot = None
            if len(self.mesh.point_data.keys()) > 0:
                to_plot = list(self.mesh.point_data.keys())[0]
            elif len(self.mesh.cell_data.keys()) > 0:
                to_plot = list(self.mesh.point_data.keys())[0]
            if to_plot is not None:
                self.set_mesh_display(to_plot)
                
        self.plotter.show()
    
    def add_point_data(self,point_data_dict:Optional[dict] = None,**kwargs):
        if isinstance(point_data_dict,dict):
            for key,val in point_data_dict.items():
                self.mesh.point_data[key] = val
        
        for key,val in kwargs.items():
            self.mesh.point_data[key] = val
                
    def add_cell_data(self,point_data_dict:Optional[dict] = None,**kwargs):
        if isinstance(point_data_dict,dict):
            for key,val in point_data_dict.items():
                self.mesh.cell_data[key] = val
        
        for key,val in kwargs.items():
            self.mesh.cell_data[key] = val
    
    
    def add_chart(self,subplot,scalar,p1,p2,axis,resolution,label,data_type = 'point'):
        self.plotter.subplot(*subplot)
        
        chart = pv.Chart2D()
         
        self.lines[subplot] = {}
        
        line = self.mesh.sample_over_line(p1,p2,resolution= resolution)
        
        if data_type == 'point':
            y = line.point_data[scalar]
        else:
            y = line.cell_data[scalar]
            data_type = 'cell'
        
        
        x = line.points[:,axis]
        line = chart.line(x,y,label = label)
        
        self.lines[subplot][label] = Line(scalar,p1,p2,axis,label,resolution,data_type,line)
        
        self.charts[subplot] = chart
        self.plotter.add_chart(self.charts[subplot])
        
    
    def add_data_to_chart(self,subplot,x,y,label,**kwargs):
        '''
        Add immutable data to a chart. This does not get updated. Use for Benchmark Data
        '''
        assert isinstance(self.charts[subplot],pv.Chart2D)
        self.charts[subplot].line(x,y,label = label,**kwargs)
         
    
    def set_animation(self,name,framerate = 24):
        self.plotter.show(interactive_update= True)
        self.plotter.open_movie(name,framerate)
        
    
    def update_frame(self):
        for key in self.charts.keys():
            lines = self.lines[key]
            for label in lines.keys():
                line = lines[label]
                sample_line = self.mesh.sample_over_line(line.p1,line.p2,resolution=line.resolution)
                if line.data_type == 'point':
                    y = sample_line.point_data[line.scalar]
                else:
                    y = sample_line.cell_data[line.scalar]
                    
                x = sample_line.points[:,line.axis]
                line.pv_line.update(x,y)
        
        self.plotter.write_frame()
        
    
    def close(self):
        self.plotter.close()
                        
                